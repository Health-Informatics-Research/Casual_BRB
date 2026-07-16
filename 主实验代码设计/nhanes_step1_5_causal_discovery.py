# =============================================================================
# nhanes_step1_5_causal_discovery.py 【诊断修复版】
# 修复点：
# 1. 采用 G-square 检验替代 Fisher Z (适配混合/离散数据)
# 2. 放宽背景知识：仅禁止 Target -> Features，允许特征间自由学习
# 3. 引入 Mutual Information (MI) 加权投票，替代简单的多数表决
# 4. 剔除无通路变量，防止纯噪声注入 BRB
# =============================================================================

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder

# 导入因果发现库
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import gsq  # 改用 G 检验
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 0. 配置 (保持与原版一致的代表变量)
# =============================================================================
PATHWAY_REPRESENTATIVES = {
    "P1_Socioeconomic": ["INDFMPIR", "DMDEDUC2"],
    "P2_Sleep": ["SLD012", "SLQ050"],
    "P3_HealthStatus": ["HUQ010", "BMXBMI"],
    "P4_FoodSecurity": ["FSDAD", "FSDHH"],
    "P5_Clinical": ["BPQ020", "MCQ160A", "MCQ080", "DIQ010"],
    "P6_Substance": ["SMQ020", "ALQ130", "DUQ200"],
}

TARGET_VAR = "depression_label"

# =============================================================================
# 1. 数据准备
# =============================================================================
def load_and_prepare_data():
    print("加载清洗后的原始数据...")
    df = pd.read_csv("nhanes_clean_original.csv")
    all_rep_vars = []
    for path, vars in PATHWAY_REPRESENTATIVES.items():
        for v in vars:
            if v in df.columns:
                all_rep_vars.append(v)
            else:
                print(f"警告：变量 {v} 不在数据中，已跳过")
                
    analysis_vars = all_rep_vars + [TARGET_VAR]
    df_sub = df[analysis_vars].copy()
    
    # 填补中位数以防万一（理论上Step1已插补，但MI要求无NaN）
    df_sub = df_sub.fillna(df_sub.median())
    print(f"最终纳入因果发现的变量 (共 {len(analysis_vars)} 个): {analysis_vars}")
    return df_sub, all_rep_vars

# =============================================================================
# 2. PC 算法 (离散化 + G-square + 宽松BK)
# =============================================================================
def run_pc_with_bk(data, node_names, sink_node):
    print("\n" + "="*60)
    print("开始运行 PC 算法 (G-square + 分层时间先验)")
    print("="*60)
    
    # 离散化代码保持不变
    X_encoded = np.zeros_like(data.values)
    from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
    for i, col in enumerate(data.columns):
        if len(data[col].unique()) > 10:
            est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            X_encoded[:, i] = est.fit_transform(data[[col]]).flatten()
        else:
            est = OrdinalEncoder()
            X_encoded[:, i] = est.fit_transform(data[[col]]).flatten()
            
    bk = BackgroundKnowledge()
    cg_nodes = [GraphNode(name) for name in node_names]
    idx_map = {name: i for i, name in enumerate(node_names)}
    
    # 【核心新增】：定义流行病学时间分层 (Tier 1 发生最早)
    tiers = {
        1: ["INDFMPIR", "DMDEDUC2"],                                # 背景层 (社会经济)
        2: ["SLD012", "SLQ050", "FSDAD", "FSDHH", "SMQ020", "ALQ130", "DUQ200"], # 行为/环境层
        3: ["HUQ010", "BMXBMI", "BPQ020", "MCQ160A", "MCQ080"],     # 临床/生理层
        4: [sink_node]                                              # 结果层 (抑郁)
    }
    
    # 将变量映射到对应的 Tier
    node_tier_map = {}
    for tier, vars_in_tier in tiers.items():
        for v in vars_in_tier:
            if v in node_names:
                node_tier_map[v] = tier
                
    # 添加约束：高 Tier (发生较晚) 绝对不能指向 低 Tier (发生较早)
    for i, node_i in enumerate(node_names):
        for j, node_j in enumerate(node_names):
            tier_i = node_tier_map.get(node_i, 99)
            tier_j = node_tier_map.get(node_j, 99)
            # 如果 node_i 发生的时间比 node_j 晚，则禁止 i -> j
            if tier_i > tier_j:
                bk.add_forbidden_by_node(cg_nodes[i], cg_nodes[j])
                
    print("  已加载流行病学分层先验：禁止逆时间因果边。")
    
    # 运行 PC
    cg = pc(X_encoded, alpha=0.10, indep_test=gsq, stable=True,
            node_names=node_names, background_knowledge=bk, show_progress=False)
    
    adj = cg.G.graph
    edges = []
    print("\n发现的有向边 (符合时间逻辑):")
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if adj[i, j] == -1 and adj[j, i] == 1:
                edges.append((node_names[i], node_names[j]))
                print(f"  {node_names[i]} --> {node_names[j]}")
                
    return cg, edges
# =============================================================================
# 3. 距离计算与 MI 加权层级聚类
# =============================================================================
def build_dag_and_roles(node_names, edges):
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    G.add_edges_from(edges)
    
    sink = TARGET_VAR
    distances = {}
    for node in node_names:
        if node == sink:
            distances[node] = 0
        else:
            try:
                distances[node] = nx.shortest_path_length(G, source=node, target=sink)
            except nx.NetworkXNoPath:
                distances[node] = np.inf
                
    # 判定角色
    roles = {}
    for node in node_names:
        if node == sink: continue
        if np.isinf(distances[node]):
            roles[node] = 'none'
        elif distances[node] == 1:
            roles[node] = 'direct'
        else:
            # 距离 >= 2 时，判断是否为根节点 (入度为0)
            if G.in_degree(node) == 0:
                roles[node] = 'root'
            else:
                roles[node] = 'interm'
                
    return G, distances, roles

def cluster_pathways_weighted(df, all_rep_vars, roles_dict):
    print("\n" + "="*60)
    print("基于互信息 (MI) 的加权路径聚类")
    print("="*60)
    
    # 计算每个变量与目标变量的互信息 (MI) 作为权重
    X = df[all_rep_vars]
    y = df[TARGET_VAR]
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_weights = dict(zip(all_rep_vars, mi_scores))
    
    sub1, sub2, sub3, excluded = [], [], [], []
    
    for path_name, vars in PATHWAY_REPRESENTATIVES.items():
        role_scores = {'root': 0.0, 'direct': 0.0, 'interm': 0.0}
        has_valid_path = False
        
        print(f"\n评估路径: {path_name}")
        for v in vars:
            if v not in df.columns: continue
            role = roles_dict.get(v, 'none')
            weight = mi_weights.get(v, 0)
            print(f"  - 变量 {v}: 角色={role}, MI权重={weight:.4f}")
            
            if role != 'none':
                role_scores[role] += weight
                has_valid_path = True
                
        # 修复点 B & D：无通路变量直接剔除，不强加给 Sub3
        if not has_valid_path:
            print(f"  [!] 警告: {path_name} 所有代表变量均无通路，排除出 BRB")
            excluded.append(path_name)
            continue
            
        # 多数表决改为加权打分
        majority_role = max(role_scores, key=role_scores.get)
        print(f"  => 聚合得分: Root({role_scores['root']:.4f}), Direct({role_scores['direct']:.4f}), Interm({role_scores['interm']:.4f})")
        print(f"  => 最终定级: {majority_role}")
        
        if majority_role == 'root': sub1.append(path_name)
        elif majority_role == 'direct': sub2.append(path_name)
        elif majority_role == 'interm': sub3.append(path_name)
        
    return {"Sub1": sub1, "Sub2": sub2, "Sub3": sub3}, excluded

# =============================================================================
# 主流程
# =============================================================================
def main():
    print("="*70)
    print("NHANES 因果发现与加权数据驱动层级映射 (诊断修复版)")
    print("="*70)
    
    df, all_rep_vars = load_and_prepare_data()
    node_names = all_rep_vars + [TARGET_VAR]
    
    # 运行修复后的 PC
    cg, edges = run_pc_with_bk(df, node_names, TARGET_VAR)
    
    # 构建图并获取角色
    G, distances, roles = build_dag_and_roles(node_names, edges)
    
    # 加权路径聚类
    sub_mapping, excluded = cluster_pathways_weighted(df, all_rep_vars, roles)
    
    print("\n" + "="*60)
    print("最终 BRB 层级映射结果：")
    print(f"  Sub1_background (根源层): {sub_mapping['Sub1']}")
    print(f"  Sub2_behavioral (行为层): {sub_mapping['Sub2']}")
    print(f"  Sub3_stress     (压力层): {sub_mapping['Sub3']}")
    print(f"  被剔除的纯噪声路径: {excluded}")
    print("="*60)
    
    # 保存给 Step 2 用的配置文件
    brb_config = {
        "Sub1_background": sub_mapping["Sub1"],
        "Sub2_behavioral": sub_mapping["Sub2"],
        "Sub3_stress": sub_mapping["Sub3"]
    }
    np.save("brb_hierarchy_config.npy", brb_config)
    print("\nBRB 层级配置已更新保存至 brb_hierarchy_config.npy")

if __name__ == "__main__":
    main()