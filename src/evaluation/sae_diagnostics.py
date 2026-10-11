"""Descriptive correlations and overlap measures; no semantic concept claims."""
import numpy as np
from scipy.stats import spearmanr


def correlation(a,b,rank=False):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:return None
    return float(spearmanr(a,b).statistic if rank else np.corrcoef(a,b)[0,1])


def residualize(values,controls):
    design=np.column_stack([np.ones(len(values)),controls])
    return values-design@np.linalg.lstsq(design,values,rcond=None)[0]


def top_overlap(a,b,count):
    count=min(count,len(a))
    a,b=np.asarray(a),np.asarray(b)
    left=set(i for i in np.argsort(-a)[:count] if a[i]>0)
    right=set(i for i in np.argsort(-b)[:count] if b[i]>0)
    denominator=min(len(left),len(right))
    return len(left&right)/denominator if denominator else None


def active_jaccard(a,b):
    left,right=np.asarray(a)>0,np.asarray(b)>0
    union=(left|right).sum()
    return float((left&right).sum()/union) if union else None
