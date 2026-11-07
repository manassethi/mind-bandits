"""Gap-contraction sensitivity check; run from the project root.

Uses existing cached environments, never retrains or overwrites previous results.
"""
from pathlib import Path
import sys
import os
import json
import numpy as np
from scipy import stats
ROOT=Path(os.environ.get("CS592_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0,str(ROOT))
from src.bandits.environment import BanditEnv
from src.bandits.policies import policy_suite
from src.bandits.runner import run
EVIDENCE=ROOT/'latex'/'evidence'
def read(path): return json.loads((ROOT/path).read_text())
def ci(x):
    x=np.asarray(x,float)
    return [float(x.mean()),float(stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))]
def experiment():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    fams=['A/B test (tuned length)','UCB1','Thompson sampling']
    audit={'gap_sensitivity':{}}
    # Independently rerun a short-horizon gap contraction experiment.
    tuned=read("results/q1/tuned.json")["K10_T10k"]
    with np.load(ROOT/"results/q1/envs/test_K10_T10k.npz") as z:
        P0=z["P"]; articles=z["articles"]; seed=int(z["seed"])
    baseP=P0.mean(2,keepdims=True)
    selected={f:tuned[f] for f in fams[:3]}
    selected["Uniform"]="uniform"
    arrays={}
    for lam in [0.,.25,.5,1.]:
        P=baseP + lam*(P0-baseP)
        assert np.all((P>=0)&(P<=1))
        assert np.allclose(P.mean(2), P0.mean(2), atol=1e-7)
        env=BanditEnv(P,articles,10000,seed=seed)
        policies={p.name:p for p in policy_suite(env.S,env.K,env.horizon,env.mu0.mean(1),env.seed)}
        row={}
        for fam,name in selected.items():
            res=run(policies[name],env)
            r=res["regret"][:,-1]
            if lam==0: assert np.allclose(r,0,atol=.001)
            arrays[f"lambda{lam}/{fam}"]=r
            row[fam]={"variant":name,"regret":ci(r),"regret_runs":r.tolist()}
        ref=np.array(row["Thompson sampling"]["regret_runs"])
        for fam in selected:
            row[fam]["difference_vs_ts"]=ci(np.array(row[fam]["regret_runs"])-ref)
        audit["gap_sensitivity"][str(lam)]=row
        print("gap scale",lam,{f: [round(x,2) for x in v["regret"]] for f,v in row.items()},flush=True)
    # Check that lambda=1 exactly reconstructs the cached short-horizon results.
    with np.load(ROOT/"results/q1/test/K10_T10k.npz") as z:
        for fam,v in selected.items():
            np.testing.assert_allclose(arrays[f"lambda1.0/{fam}"],z[f"{v}/regret"][:,-1],atol=.02,rtol=1e-4)
    audit["gap_sensitivity_protocol"]={"horizon":10000,"runs":30,"K":10,
        "formula":"P_lambda[u,a] = mean_a P[u,a] + lambda * (P[u,a] - mean_a P[u,a])",
        "parameters":"Frozen at train-tuned K10_T10k settings; no retuning under contraction",
        "checks":"probability bounds, conserved user mean, zero-gap regret, lambda=1 cached-run replication"}
    np.savez(EVIDENCE/"gap_sensitivity.npz",**arrays)
    (EVIDENCE/"gap_sensitivity.json").write_text(json.dumps(audit,indent=2)+"\n")
    return audit

if __name__=="__main__": experiment()
