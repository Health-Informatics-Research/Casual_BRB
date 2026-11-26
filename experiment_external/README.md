# 跨数据集实验修订

本模块独立于已冻结的 experiment_v3 和 exploratory_tuning_v1。不会修改原主实验、两个旧外部数据目录或论文数字。默认仅审计；训练需显式执行。

## 最终方案与论文定位

- 两个数据集的主补充实验均为**独立重训的方法适用性验证**，不是原 NHANES 2015–2018 拟合模型的直接泛化验证。沿用主模型推理、DE、分层和单调规则表机制。
- NHANES 2011–2014 使用原始 XPT 重建成人、完整 PHQ-9、阈值10的队列。睡眠时长保留原名 SLD010H，不将它冒充 SLD012；缺失率筛选、插补均在 Fit 或相应内部训练折拟合。
- 另提供严格的原模型 transport 入口；逐周期检查原处理器要求的字段。原模型需要的 SLD012、SLQ120、SLQ040 等缺失时，入口拒绝预测，不将整列填空后宣称有效外部验证。没有新增共同特征模型或重新选择主模型。
- MIDUS 使用 B1PDEPDX 的原生抑郁组合标签，保留变量原名和量表。P4 是财务压力，不写成食物安全。标签不是 PHQ-9，AP/Brier 的绝对值不能直接跨数据集解释为性能提升。
- 不使用 B1SCHRON/B1SCHROX：构念文档 PDF 第111页说明它们汇总 SAQ A11a–dd，含 A11T 的焦虑/抑郁项目。P5 改用三个明确的躯体疾病字段。
- MIDUS 成人需有效标签/年龄并参加 SAQ。逐变量处理正、负特殊编码；不把拒答98当睡眠小时数，不生成假调查权重。B1PA39 的结构性跳题作为缺失处理，不直接改成“不吸烟”；可能被 Fit 缺失率规则剔除。
- MIDUS 按 M2FAMNUM 隔离家庭：外层25折分配成 Test5/Cal4/Val4/Fit12折，比例为近似20/16/16/48。OOF、BRB内部CV、ML内部CV均用家庭分组分层折；Test置信区间以家庭为重采样单位。其余 NHANES 使用原有参与者分层划分与参与者bootstrap。
- 五种子42/99/123/2026/888；21配置；所有DE模型统一240代、28个体，其余参数与冻结core一致；权重固定0.45/0.25/0.30。所有模型统一Cal正斜率Platt、Val网格选F2阈值，外部数据上不重新比较校准方案或预算。
- 完整Fit结构在内部CV外固定，依旧不是完全嵌套验证。NHANES 保留 DIQ010 层99以对应原实现；MIDUS各原生代表变量的时间层在profile中明示，属于新队列的建模假设。
- 旧外部结果已经查看过，重跑应写成冻结协议下的探索性跨数据集适用性实验。不能恢复旧论文“无偏ATE”“外部验证优越”的结论。

## 执行

在项目目录，使用 D:\Python\python.exe（原实验环境），不要用文档插件的Python加载旧模型。

```powershell
& .\Start-ExternalValidation.ps1 -Stage audit -Dataset nhanes
& .\Start-ExternalValidation.ps1 -Stage audit -Dataset midus
& 'D:\Python\python.exe' -m experiment_external.checks

# 正式训练：全五划分，依次运行，不自动并行占满内存
& .\Start-ExternalValidation.ps1 -Stage train -Dataset nhanes
& .\Start-ExternalValidation.ps1 -Stage train -Dataset midus

# 或指定一个划分，仍使用同一冻结协议，可断点续跑
& .\Start-ExternalValidation.ps1 -Stage train -Dataset midus -Seed 42

# 当前NHANES字段不兼容，预期在预测前拒绝；不要把失败当作模型效果差
& .\Start-ExternalValidation.ps1 -Stage transport -Dataset nhanes
```

入口将 Python/joblib 临时目录定向到D盘；不会承诺Windows自身和应用程序完全不写C盘。

## 输出与复现

独立输出 `runs/external_v1_*`。首次训练保存协议、profile、源码快照、源码/输入文件SHA256、软件版本；续跑变化即停止。每种子保存划分、Fit字段审计、结构、内部验证参与者、路径处理器、模型、Cal/Val/Test预测、校准器、阈值、DE日志及优化状态。BRB预测包含Bel_D、Pl_D、m_Theta。逐模型有完成标记，已完成模型不再训练/测试；意外中断可用完全相同配置恢复。

`metric_CI.csv` 为九指标95%区间，`paired_differences.csv` 为与V0、V1′、BRB-DR、LR两输入版本的配对差。`summary.csv` 自动给均值/SD/已完成划分数，`summary_status.json` 明示是否完成五划分。不把部分结果当完整实验，不把跨划分SD当独立重复的标准误。

本次主范围为预测性能与可追溯性；不自动追加新的干预、不确定性增量或调查加权结论。MIDUS按家庭bootstrap，不能照抄主实验的参与者bootstrap描述。后续论文应新增跨数据集适用性小节，并按实际字段数、样本量、标签定义、分组和结果填写。
