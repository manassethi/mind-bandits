"""Contextual bandits with policy-unseen articles introduced halfway through a run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.bandits.contextual import LinUCB, LinearThompson, AvailableBandit
from src.evaluation.sweep import interval


def available_articles(pool, initial_count, round_index, cold_start_round):
    return np.asarray(pool[:initial_count] if round_index<cold_start_round else pool,dtype=int)


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment',type=Path,default=Path('dataset/processed/q2_environment.npz'))
    p.add_argument('--rounds',type=int,default=10000)
    p.add_argument('--seeds',type=int,default=20)
    p.add_argument('--pool-size',type=int,default=20)
    p.add_argument('--candidates',type=int,default=5)
    p.add_argument('--cold-start-round',type=int,default=None)
    p.add_argument('--cold-window',type=int,default=1000)
    p.add_argument('--alpha',type=float,default=0.1)
    p.add_argument('--ts-scale',type=float,default=0.1)
    p.add_argument('--epsilon',type=float,default=0.1)
    p.add_argument('--seed',type=int,default=100)
    p.add_argument('--pool-seed',type=int,default=2026)
    p.add_argument('--output-dir',type=Path,default=Path('results/q2'))
    args=p.parse_args()
    if not args.environment.exists():p.error('Run python scripts/prepare_contextual.py first')
    data=np.load(args.environment,allow_pickle=False)
    x=data['features'];prob=data['probabilities'];meta=json.loads(str(data['metadata']))
    if x.ndim!=3 or prob.shape!=x.shape[:2] or not np.all(np.isfinite(x)) or not np.all(np.isfinite(prob)) or np.any((prob<0)|(prob>1)):
        p.error('Invalid contextual environment arrays')
    cold=args.cold_start_round if args.cold_start_round is not None else args.rounds//2
    initial=args.pool_size//2
    if args.seeds<2 or not 0<cold<args.rounds or not 1<=args.cold_window<=args.rounds-cold:p.error('Need >=2 seeds and a cold window fitting after introduction')
    if not 4<=args.pool_size<=x.shape[1] or not 1<=args.candidates<=initial:p.error('Need 4 <= pool size <= article bank, and candidates <= half pool size')
    if args.alpha<0 or args.ts_scale<0 or not 0<=args.epsilon<=1:p.error('Invalid exploration parameters')
    pool=np.random.default_rng(args.pool_seed).choice(x.shape[1],args.pool_size,replace=False).tolist()
    names=['LinUCB category','LinUCB history','Linear Thompson history','Epsilon greedy','UCB1','Thompson sampling']
    trajectories={name:[] for name in names};rows=[]
    for seed in range(args.seed,args.seed+args.seeds):
        print(f'Seed {seed-args.seed+1}/{args.seeds}...',flush=True)
        schedule_rng=np.random.default_rng(np.random.SeedSequence([seed,0]))
        users=schedule_rng.integers(x.shape[0],size=args.rounds)
        candidates=[schedule_rng.choice(available_articles(pool,initial,t,cold),args.candidates,replace=False) for t in range(args.rounds)]
        uniforms=np.random.default_rng(np.random.SeedSequence([seed,1])).random((args.rounds,args.candidates))
        for index,name in enumerate(names):
            rng=np.random.default_rng(np.random.SeedSequence([seed,2,index]))
            if name=='LinUCB category':policy=LinUCB(meta['category_dimensions'],rng,args.alpha)
            elif name=='LinUCB history':policy=LinUCB(x.shape[2],rng,args.alpha)
            elif name=='Linear Thompson history':policy=LinearThompson(x.shape[2],rng,args.ts_scale)
            else:policy=AvailableBandit(x.shape[1],rng,{'Epsilon greedy':'Epsilon','UCB1':'UCB1','Thompson sampling':'Thompson'}[name],args.epsilon)
            rewards=np.zeros(args.rounds);regrets=np.zeros(args.rounds);new_selected=np.zeros(args.rounds);new_opportunity=np.zeros(args.rounds)
            for t,arms in enumerate(candidates):
                features=x[users[t],arms]
                if name=='LinUCB category':features=features[:,:meta['category_dimensions']]
                contextual=index<3
                chosen=policy.select(features if contextual else arms)
                reward=int(uniforms[t,chosen]<prob[users[t],arms[chosen]])
                policy.update(features[chosen] if contextual else arms[chosen],reward)
                rewards[t]=reward;regrets[t]=prob[users[t],arms].max()-prob[users[t],arms[chosen]]
                new_selected[t]=arms[chosen] in pool[initial:]
                new_opportunity[t]=any(a in pool[initial:] for a in arms)
            trajectories[name].append((rewards.cumsum(),regrets.cumsum()))
            for phase,start,end in [('all',0,args.rounds),('before',0,cold),('after',cold,args.rounds),('cold_window',cold,cold+args.cold_window)]:
                rows.append({'seed':seed,'policy':name,'phase':phase,'rounds':end-start,'clicks':int(rewards[start:end].sum()),
                             'ctr':float(rewards[start:end].mean()),'regret':float(regrets[start:end].sum()),
                             'regret_per_round':float(regrets[start:end].mean()),'new_article_choices':int(new_selected[start:end].sum()),
                             'new_article_opportunities':int(new_opportunity[start:end].sum())})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.output_dir/'runs.csv',rows)
    summaries=[];paired=[]
    for phase in ['all','before','after','cold_window']:
        for name in names:
            block=[r for r in rows if r['policy']==name and r['phase']==phase]
            entry={'policy':name,'phase':phase}
            for metric in ['clicks','ctr','regret','regret_per_round']:
                mean,low,high=interval([r[metric] for r in block]);entry.update({metric+'_mean':mean,metric+'_ci95_low':low,metric+'_ci95_high':high})
            summaries.append(entry)
            if name=='Thompson sampling':continue
            baseline=[r for r in rows if r['policy']=='Thompson sampling' and r['phase']==phase]
            entry={'policy':name,'baseline':'Thompson sampling','phase':phase}
            for metric,sign,label in [('clicks',1,'click_gain'),('regret',-1,'regret_reduction')]:
                mean,low,high=interval([sign*(a[metric]-b[metric]) for a,b in zip(block,baseline)])
                entry.update({label+'_mean':mean,label+'_ci95_low':low,label+'_ci95_high':high})
            paired.append(entry)
    write_csv(args.output_dir/'summary.csv',summaries);write_csv(args.output_dir/'paired_comparisons.csv',paired)
    config={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'cold_start_round':cold,'initial_articles':pool[:initial],'new_articles':pool[initial:],'environment_metadata':meta,'environment_sha256':hashlib.sha256(args.environment.read_bytes()).hexdigest()}
    (args.output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,5));t=np.arange(1,args.rounds+1)
    for name,values in trajectories.items():
        values=np.asarray(values)
        for j,ax in enumerate(axes):ax.plot(t,values[:,j].mean(axis=0),label=name)
    for ax,label in zip(axes,['Cumulative clicks','User-specific cumulative expected regret']):
        ax.axvline(cold,color='black',linestyle='--');ax.set(xlabel='Round',ylabel=label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Q2 category/history simulator | dashed line: new articles introduced')
    fig.tight_layout();fig.savefig(args.output_dir/'comparison.png',dpi=180);plt.close(fig)
    for row in summaries:
        if row['phase']=='all':print(f"{row['policy']:25s} clicks={row['clicks_mean']:.1f} regret={row['regret_mean']:.1f}")
    print(f'Saved results to {args.output_dir}')


if __name__=='__main__':main()
