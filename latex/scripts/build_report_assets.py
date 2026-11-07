"""Rebuild milestone figures and audit the cached experiment outputs.

Run from the repository root with .venv/bin/python latex/scripts/build_report_assets.py.
The additional experiment contracts article gaps without changing each user's
mean click probability. Policies and parameters are frozen before this sweep.
Existing experiment files and the fitted click model are never overwritten.
"""
from pathlib import Path
import sys
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from src.bandits.environment import BanditEnv
from src.bandits.policies import policy_suite
from src.bandits.runner import run

OUT = ROOT / "latex" / "figures"
EVIDENCE = ROOT / "latex" / "evidence"
OUT.mkdir(parents=True, exist_ok=True)
EVIDENCE.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "axes.labelsize": 9, "legend.fontsize": 8})
COLORS = ["#30659A", "#C56A30", "#43866B", "#855C9A"]

def read(path):
    return json.loads((ROOT / path).read_text())

def ci(x):
    x = np.asarray(x, dtype=float)
    return [float(x.mean()), float(stats.t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)))]

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)

def main():
    import os
    os.chdir(ROOT)
    m = read("results/click_model/metrics.json")
    cal = read("results/click_model/calibration.json")
    q1 = read("results/q1/summary.json")
    q2 = read("results/q2/summary.json")
    audit = {"sources": {}, "q1_gaps": {}, "q2_gaps": {}, "gap_sensitivity": {}}
    sources = ["results/click_model/metrics.json", "results/click_model/calibration.json",
        "results/click_model/selection.json", "results/q1/summary.json",
        "results/q1/tuned.json", "results/q2/summary.json", "results/q2/tuned.json",
        "results/q3/results.json", "results/q4/summary.json",
        "results/q4/prior_validation.json", "results/q1/envs/test_K10.npz",
        "results/q1/envs/test_K10_T10k.npz", "results/q1/test/K10_T10k.npz",
        "results/q2/envs/test_life2000.npz"]
    for path in sources:
        audit["sources"][path] = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    # Constant predictor is fixed to the TRAIN base rate, not the evaluation labels.
    base = cal["target_ctr"]
    ctr = m["test"]["observed_ctr"]
    audit["constant_train_ctr_baseline"] = {"probability": base, "auc": .5,
        "log_loss": float(-ctr*np.log(base)-(1-ctr)*np.log1p(-base)),
        "brier": float(ctr*(1-base)**2+(1-ctr)*base**2)}
    for cond in ["K2", "K5", "K10", "K20", "K50"]:
        with np.load(ROOT / f"results/q1/envs/test_{cond}.npz") as z:
            mu = z["P"].mean(1)
        gap = np.sort(mu, axis=1)[:, -1] - np.sort(mu, axis=1)[:, -2]
        audit["q1_gaps"][cond] = {"quantiles_10_50_90": np.quantile(gap,[.1,.5,.9]).tolist(),
            "share_gap_below_0.001": float((gap<.001).mean()), "runs": len(gap)}
    # Sample 100 decision contexts per run with a separate fixed RNG.
    with np.load(ROOT / "results/q2/envs/test_life2000.npz") as z:
        P = z["P"]
    rng = np.random.default_rng(7102026)
    gaps = []
    for s in range(len(P)):
        for t,u in zip(rng.integers(0,20000,100), rng.integers(0,P.shape[1],100)):
            live = np.arange(20)
            for j in range(int(t)//100):
                live[j % 20] = 20+j
            v = np.sort(P[s,u,live])
            gaps.append(float(v[-1]-v[-2]))
    audit["q2_gaps"] = {"contexts": len(gaps),
        "quantiles_10_50_90": np.quantile(gaps,[.1,.5,.9]).tolist(),
        "share_gap_below_0.001": float((np.array(gaps)<.001).mean())}

    # Reliability plot and the actual distribution of decision gaps.
    fig, axs = plt.subplots(1,2,figsize=(6.7,2.0),layout="constrained")
    ax = axs[0]
    for bins, label, color in [(m["test_calibration_bins"],"Before anchoring",COLORS[0]),
        (cal["test_calibration_bins_after"],"After anchoring",COLORS[1])]:
        ax.plot([r["mean_pred"]*100 for r in bins], [r["observed"]*100 for r in bins],
            marker="o",ms=3,lw=1.3,label=label,color=color)
    ax.plot([0,21],[0,21],ls=":",color="0.5",lw=1)
    ax.set(xlabel="Predicted click probability (%)",ylabel="Observed CTR (%)",
        title="(a) Reliability on logged candidates",xlim=(0,21),ylim=(0,21))
    ax.legend(frameon=False,loc="upper left",fontsize=7)
    ax = axs[1]
    with np.load(ROOT / "results/q1/envs/test_K10.npz") as z:
        mu = z["P"].mean(1)
    g1=np.sort(mu,axis=1)[:,-1]-np.sort(mu,axis=1)[:,-2]
    for g,l,c in [(g1,"Q1: 30 pools",COLORS[0]),(np.array(gaps),"Q2: 3,000 contexts",COLORS[2])]:
        x=np.sort(g)*100
        ax.step(x,np.arange(1,len(x)+1)/len(x),where="post",label=l,color=c,lw=1.5)
    ax.set(xscale="log",xlabel="Best-minus-second gap (percentage points)",
        ylabel="Cumulative fraction",title="(b) Choice difficulty",ylim=(0,1.03))
    ax.legend(frameon=False,fontsize=7)
    save(fig,"simulator_validation")

    # Q1 curves: no log-axis exaggeration of early-round separation.
    fig,axs=plt.subplots(1,2,figsize=(6.7,2.5),layout="constrained")
    fams=["A/B test (tuned length)","UCB1","Thompson sampling","Discounted Thompson"]
    for ax,cond,title in zip(axs,["K10","K10_abrupt"],["(a) Stationary articles","(b) Abrupt preference switch"]):
        with np.load(ROOT/f"results/q1/test/{cond}.npz") as z:
            for family,color in zip(fams,COLORS):
                v=q1[cond]["families"][family]["variant"]
                r=z[f"{v}/regret"]
                marks=z[f"{v}/marks"]
                means=r.mean(0); half=stats.t.ppf(.975,len(r)-1)*r.std(0,ddof=1)/np.sqrt(len(r))
                ax.plot(marks/1000,means,label=family.replace("A/B test (tuned length)","A/B (tuned length)"),color=color,lw=1.5)
                ax.fill_between(marks/1000,means-half,means+half,color=color,alpha=.12,lw=0)
        if cond.endswith("abrupt"):ax.axvline(50,color="0.5",ls=":",lw=1)
        ax.set(xlabel="Recommendation rounds (thousands)",ylabel="Cumulative pseudo-regret",title=title,xlim=(0,100))
    axs[0].legend(frameon=True,facecolor="white",edgecolor="none",framealpha=.95,fontsize=7,loc="upper left")
    save(fig,"q1_progress")

    # Reuse a completed cluster run when present; otherwise run locally.
    gap_file=EVIDENCE/'gap_sensitivity.json'
    if gap_file.exists():
        audit.update(json.loads(gap_file.read_text()))
    else:
        from gap_sensitivity import experiment
        audit.update(experiment())
    (EVIDENCE/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")

if __name__=="__main__":main()
