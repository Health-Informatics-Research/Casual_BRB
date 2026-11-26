"""Exact G-square statistic with cached empirical partitions (same PC decisions)."""
from collections import OrderedDict
import numpy as np
from scipy.special import chdtrc
from causallearn.utils.cit import CIT_Base,register_ci_test

class CachedGSquare(CIT_Base):
    def __init__(self,data,**kwargs):
        encoded=np.apply_along_axis(lambda c:np.unique(c,return_inverse=True)[1],0,data).astype(np.int64)
        super().__init__(encoded,**kwargs)
        self.check_cache_method_consistent('cached_exact_gsq','same_G_statistic_and_observed_cell_degrees_of_freedom')
        self.assert_input_data_is_valid()
        self.cards=encoded.max(axis=0)+1
        self.partitions=OrderedDict(); self.distinct=OrderedDict(); self.entropies={}; self.group_sizes={}
        self.cap=65536
        self.index_dtype=np.uint16 if self.sample_size<=65535 else np.uint32
        self.count_dtype=np.uint8 if self.cards.max()<=255 else np.uint32

    def __deepcopy__(self,memo):
        # Raw observations are immutable; graph copies may safely share deterministic CI memoization.
        memo[id(self)]=self
        return self

    def partition(self,mask):
        if mask in self.partitions:
            self.partitions.move_to_end(mask); return self.partitions[mask]
        if not mask:
            result=(np.zeros(self.sample_size,dtype=self.index_dtype),np.array([0],dtype=self.index_dtype),1)
            entropy=self.sample_size*np.log(self.sample_size)
        else:
            j=mask.bit_length()-1; previous=mask^(1<<j)
            inverse,_,parent_groups=self.partition(previous)
            codes=inverse.astype(np.int64)*self.cards[j]+self.data[:,j]
            # Parent IDs are dense: counting refines the same partition as np.unique,
            # without sorting observations. Any representative of a cell is sufficient.
            full_counts=np.bincount(codes,minlength=int(parent_groups*self.cards[j]))
            observed=np.flatnonzero(full_counts)
            recode=np.zeros(len(full_counts),dtype=self.index_dtype)
            recode[observed]=np.arange(len(observed),dtype=self.index_dtype)
            representatives=np.zeros(len(full_counts),dtype=self.index_dtype)
            representatives[codes]=np.arange(self.sample_size,dtype=self.index_dtype)
            counts=full_counts[observed]
            result=(recode[codes],representatives[observed],len(counts))
            entropy=float(np.dot(counts,np.log(counts)))
        self.entropies[mask]=float(entropy); self.group_sizes[mask]=result[2]; self.partitions[mask]=result
        if len(self.partitions)>self.cap: self.partitions.popitem(last=False)
        return result

    def entropy(self,mask):
        if mask not in self.entropies: self.partition(mask)
        return self.entropies[mask]

    def n_groups(self,mask):
        if mask not in self.group_sizes: self.partition(mask)
        return self.group_sizes[mask]

    def counts_given(self,mask,j):
        key=(mask,j)
        if key in self.distinct:
            self.distinct.move_to_end(key); return self.distinct[key]
        inv,_,n=self.partition(mask)
        _,representatives,_=self.partition(mask|(1<<j))
        counts=np.bincount(inv[representatives],minlength=n).astype(self.count_dtype)
        self.distinct[key]=counts
        if len(self.distinct)>self.cap: self.distinct.popitem(last=False)
        return counts

    def __call__(self,X,Y,condition_set=None):
        xs,ys,ss,key=self.get_formatted_XYZ_and_cachekey(X,Y,condition_set)
        if key in self.pvalue_cache: return self.pvalue_cache[key]
        x,y=xs[0],ys[0]; mask=sum(1<<s for s in ss)
        ns=self.n_groups(mask)
        # If adding X or Y does not refine S, that variable is constant in every
        # observed S-cell; the standard observed-cell G-square has exactly 0 df.
        if self.n_groups(mask|(1<<x))==ns or self.n_groups(mask|(1<<y))==ns:
            self.pvalue_cache[key]=1.
            return 1.
        dof=int(np.dot(self.counts_given(mask,x).astype(np.int64)-1,self.counts_given(mask,y).astype(np.int64)-1))
        if dof==0: p=1.
        else:
            g=2*(self.entropy(mask|(1<<x)|(1<<y))+self.entropy(mask)-self.entropy(mask|(1<<x))-self.entropy(mask|(1<<y)))
            p=float(chdtrc(dof,max(g,0.)))
        self.pvalue_cache[key]=p
        return p

register_ci_test('cached_exact_gsq',CachedGSquare)

def install_exact_skeleton_shortcut():
    """Stop only when the separator union already contains every candidate.

    causal-learn's stable skeleton stores the union of successful separators.
    Once that union contains all current neighbors, further tests cannot alter
    either edge removal or the stored separator. No conditioning-depth cap is used.
    """
    import inspect
    from causallearn.utils.PCUtils import SkeletonDiscovery
    if getattr(SkeletonDiscovery,'_exact_union_shortcut',False): return
    source=inspect.getsource(SkeletonDiscovery.skeleton_discovery)
    old='                            for s in S:\n                                sepsets.add(s)'
    new=old+'\n                            if len(sepsets) == len(Neigh_x_noy):\n                                break'
    assert source.count(old)==1,'Unsupported causal-learn skeleton implementation'
    SkeletonDiscovery._original_skeleton_discovery=SkeletonDiscovery.skeleton_discovery
    exec(compile(source.replace(old,new),'<exact_separator_union_cache>','exec'),SkeletonDiscovery.__dict__)
    SkeletonDiscovery._exact_union_shortcut=True
