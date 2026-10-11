"""Compare contextual settings across fixed pools with paired simulation outcomes."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.bandits.contextual import LinUCB, LinearThompson, AvailableBandit
from src.evaluation.contextual import phase_masks
from src.evaluation.sweep import interval, make_pools
from scripts.run_contextual import available_articles, write_csv


def variants(alphas, scales, epsilons, include_text=False, include_embeddings=False, include_sae=False):
    result=[]
    for alpha in sorted(set(alphas)):
        result += [(f'LinUCB category alpha={alpha:g}', 'category', alpha),
                   (f'LinUCB history alpha={alpha:g}', 'history', alpha)]
    if include_text:
        result += [(f'LinUCB text alpha={alpha:g}', 'text', alpha) for alpha in sorted(set(alphas))]
    if include_embeddings:
        result += [(f'LinUCB embeddings alpha={alpha:g}', 'embeddings', alpha) for alpha in sorted(set(alphas))]
    if include_sae:
        result += [(f'LinUCB SAE alpha={alpha:g}', 'sae', alpha) for alpha in sorted(set(alphas))]
        result += [(f'LinUCB GPT2 dense alpha={alpha:g}', 'gpt2', alpha) for alpha in sorted(set(alphas))]
    result += [(f'Linear Thompson scale={scale:g}', 'linear_ts', scale) for scale in sorted(set(scales))]
    result += [(f'Epsilon={eps:g}', 'Epsilon', eps) for eps in sorted(set(epsilons))]
    return result + [('UCB1', 'UCB1', 0.), ('Thompson sampling', 'Thompson', 0.)]


def summarize(rows, group_fields, aggregate_pools=False):
    groups=defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in group_fields)].append(row)
    output=[]
    for key, block in groups.items():
        entry=dict(zip(group_fields,key))
        for metric in ['clicks','ctr','regret','regret_per_round']:
            if aggregate_pools:
                by_seed=defaultdict(list)
                for row in block: by_seed[row['seed']].append(row[metric])
                values=[np.mean(v) for v in by_seed.values()]
            else: values=[r[metric] for r in block]
            mean,low,high=interval(values)
            entry.update({metric+'_mean':mean,metric+'_ci95_low':low,metric+'_ci95_high':high})
        entry['mean_evaluated_rounds']=float(np.mean([r['evaluated_rounds'] for r in block]))
        output.append(entry)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--include-sae',action='store_true',help='Compare sparse GPT-2 features with other representations')
    p.add_argument('--include-embeddings',action='store_true',help='Compare pretrained and TF-IDF features under frozen rewards')
    p.add_argument('--include-text',action='store_true',help='Add latent-text LinUCB and paired category/history comparisons')
    p.add_argument('--environment',type=Path,default=Path('dataset/processed/q2_environment.npz'))
    p.add_argument('--pools',type=int,default=5)
    p.add_argument('--pool-size',type=int,default=20)
    p.add_argument('--rounds',type=int,default=10000)
    p.add_argument('--seeds',type=int,default=20)
    p.add_argument('--candidates',type=int,default=5)
    p.add_argument('--cold-start-round',type=int,default=None)
    p.add_argument('--cold-window',type=int,default=1000)
    p.add_argument('--alphas',type=float,nargs='+',default=[0.03,0.1,0.3])
    p.add_argument('--ts-scales',type=float,nargs='+',default=[0.03,0.1,0.3])
    p.add_argument('--epsilons',type=float,nargs='+',default=[0.05,0.1])
    p.add_argument('--seed',type=int,default=100)
    p.add_argument('--pool-seed',type=int,default=2026)
    p.add_argument('--output-dir',type=Path,default=Path('results/q2_sweep'))
    args=p.parse_args()
    if not args.environment.exists():p.error('Run python scripts/prepare_contextual.py first')
    with np.load(args.environment,allow_pickle=False) as data:
        x=data['features'];prob=data['probabilities'];meta=json.loads(str(data['metadata']))
    if x.ndim!=3 or prob.shape!=x.shape[:2] or not np.all(np.isfinite(x)) or not np.all(np.isfinite(prob)) or np.any((prob<0)|(prob>1)):
        p.error('Invalid environment arrays')
    cold=args.cold_start_round if args.cold_start_round is not None else args.rounds//2
    initial=args.pool_size//2
    if args.seeds<2 or not 0<cold<args.rounds or not 1<=args.cold_window<=args.rounds-cold:p.error('Need >=2 seeds and a valid cold-start window')
    if args.pool_size<4 or not 1<=args.candidates<=initial:p.error('Need >=4 articles per pool and candidates <= half pool size')
    if args.pools>1 and args.pool_size>=x.shape[1]:p.error('Use a larger article bank than pool size to vary pools')
    if any(not np.isfinite(v) or v<0 for v in args.alphas+args.ts_scales) or any(not np.isfinite(v) or not 0<=v<=1 for v in args.epsilons):p.error('Invalid exploration parameters')
    try:pools=make_pools(meta['articles'],args.pools,args.pool_size,args.pool_seed)
    except ValueError as e:p.error(str(e))
    if args.include_text and 'history_dimensions' not in meta:p.error('Use a text environment from prepare_text_simulator.py')
    if args.include_sae:
        if 'sae_start' not in meta:p.error('Run prepare_sae.py first')
        args.include_embeddings=True
    if args.include_embeddings:
        if 'embedding_start' not in meta:p.error('Run prepare_embeddings.py first')
        args.include_text=True
    history_dim=meta.get('history_dimensions',x.shape[2])
    choices=variants(args.alphas,args.ts_scales,args.epsilons,args.include_text,args.include_embeddings,args.include_sae)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'cold_start_round':cold,'pools':[{'initial':pool[:initial],'new':pool[initial:]} for pool in pools],
            'environment_metadata':meta,'environment_sha256':hashlib.sha256(args.environment.read_bytes()).hexdigest(),
            'interval_method':'95% Student t across seed-level averages over fixed pools; pointwise, not multiplicity-adjusted'}
    (args.output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    runs=[]
    family_ids={'category':1,'history':2,'linear_ts':3,'Epsilon':4,'UCB1':5,'Thompson':6,'text':7,'embeddings':8,'sae':9,'gpt2':10}
    for pool_id,pool in enumerate(pools):
        new=np.asarray(pool[initial:])
        for seed in range(args.seed,args.seed+args.seeds):
            print(f'Pool {pool_id+1}/{args.pools}, seed {seed-args.seed+1}/{args.seeds}...',flush=True)
            schedule=np.random.default_rng(np.random.SeedSequence([seed,pool_id,0]))
            users=schedule.integers(x.shape[0],size=args.rounds)
            candidates=np.array([schedule.choice(available_articles(pool,initial,t,cold),args.candidates,replace=False) for t in range(args.rounds)])
            uniforms=np.random.default_rng(np.random.SeedSequence([seed,pool_id,1])).random(candidates.shape)
            masks=phase_masks(candidates,new,cold,args.cold_window)
            for name,family,setting in choices:
                rng=np.random.default_rng(np.random.SeedSequence([seed,pool_id,2,family_ids[family]]))
                if family in ['category','history','text','embeddings','sae','gpt2']:
                    dim=meta['category_dimensions'] if family=='category' else history_dim if family=='history' else meta['gpt2_dimensions'] if family=='gpt2' else meta['sae_dimensions'] if family=='sae' else meta['embedding_dimensions'] if family=='embeddings' else meta.get('embedding_start',x.shape[2])
                    policy=LinUCB(dim,rng,setting)
                elif family=='linear_ts':policy=LinearThompson(history_dim,rng,setting)
                else:policy=AvailableBandit(x.shape[1],rng,family,setting)
                rewards=np.zeros(args.rounds);regrets=np.zeros(args.rounds);selected_new=np.zeros(args.rounds,dtype=bool)
                contextual=family in ['category','history','linear_ts','text','embeddings','sae','gpt2']
                for t,arms in enumerate(candidates):
                    features=x[users[t],arms]
                    if family=='category':features=features[:,:meta['category_dimensions']]
                    elif family in ['history','linear_ts']:features=features[:,:history_dim]
                    elif family=='text':features=features[:,:meta.get('embedding_start',x.shape[2])]
                    elif family=='embeddings':features=np.concatenate([features[:,:history_dim],features[:,meta['embedding_start']:meta.get('embedding_end',x.shape[2])]],axis=1)
                    elif family=='gpt2':features=np.concatenate([features[:,:history_dim],features[:,meta['gpt2_start']:meta['gpt2_end']]],axis=1)
                    elif family=='sae':features=np.concatenate([features[:,:history_dim],features[:,meta['sae_start']:]],axis=1)
                    selected=policy.select(features if contextual else arms)
                    reward=int(uniforms[t,selected]<prob[users[t],arms[selected]])
                    policy.update(features[selected] if contextual else arms[selected],reward)
                    rewards[t]=reward;regrets[t]=prob[users[t],arms].max()-prob[users[t],arms[selected]]
                    selected_new[t]=arms[selected] in new
                for phase,mask in masks.items():
                    if not mask.any():continue
                    runs.append({'pool':pool_id,'seed':seed,'policy':name,'phase':phase,'evaluated_rounds':int(mask.sum()),
                                 'clicks':int(rewards[mask].sum()),'ctr':float(rewards[mask].mean()),
                                 'regret':float(regrets[mask].sum()),'regret_per_round':float(regrets[mask].mean()),
                                 'new_article_choices':int(selected_new[mask].sum())})
            # Keep completed seeds reviewable during long runs.
            write_csv(args.output_dir/'runs.csv',runs)
    per_pool=summarize(runs,['pool','policy','phase'])
    aggregate=summarize(runs,['policy','phase'],aggregate_pools=True)
    write_csv(args.output_dir/'per_pool_summary.csv',per_pool)
    write_csv(args.output_dir/'summary.csv',aggregate)
    lookup={(r['pool'],r['seed'],r['phase'],r['policy']):r for r in runs}
    baselines=[name for name,family,_ in choices if family in ['Epsilon','Thompson']]
    if args.include_text:
        baselines += [name for name,family,_ in choices if family in ['category','history']]
    if args.include_embeddings:
        baselines += [name for name,family,_ in choices if family=='text']
    if args.include_sae:
        baselines += [name for name,family,_ in choices if family in ['embeddings','gpt2']]
    paired=defaultdict(lambda:defaultdict(list))
    for row in runs:
        for baseline in baselines:
            if row['policy']==baseline:continue
            b=lookup[row['pool'],row['seed'],row['phase'],baseline]
            key=(row['policy'],baseline,row['phase'])
            paired[key][row['seed']].append({'click_gain':row['clicks']-b['clicks'],
                                             'regret_reduction':b['regret']-row['regret'],
                                             'ctr_gain':row['ctr']-b['ctr'],
                                             'regret_per_round_reduction':b['regret_per_round']-row['regret_per_round']})
    comparisons=[]
    for (name,baseline,phase),by_seed in paired.items():
        entry={'policy':name,'baseline':baseline,'phase':phase}
        for metric in ['click_gain','regret_reduction','ctr_gain','regret_per_round_reduction']:
            mean,low,high=interval([np.mean([r[metric] for r in values]) for values in by_seed.values()])
            entry.update({metric+'_mean':mean,metric+'_ci95_low':low,metric+'_ci95_high':high})
        comparisons.append(entry)
    write_csv(args.output_dir/'paired_comparisons.csv',comparisons)
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    for ax,phase in zip(axes,['all','cold_new_offered']):
        block=sorted([r for r in aggregate if r['phase']==phase],key=lambda r:r['regret_per_round_mean'])
        positions=np.arange(len(block));mean=np.array([r['regret_per_round_mean'] for r in block])
        low=np.array([r['regret_per_round_ci95_low'] for r in block]);high=np.array([r['regret_per_round_ci95_high'] for r in block])
        ax.errorbar(mean,positions,xerr=[mean-low,high-mean],fmt='o',capsize=3)
        ax.set_yticks(positions,[r['policy'] for r in block],fontsize=8);ax.invert_yaxis()
        ax.set(xlabel='Expected regret per evaluated round (lower is better)',title=phase.replace('_',' '));ax.grid(axis='x',alpha=.2)
    fig.suptitle(f'Q2 sweep | {args.pools} fixed pools | 95% t intervals across {args.seeds} seeds')
    fig.tight_layout();fig.savefig(args.output_dir/'comparison.png',dpi=180);plt.close(fig)
    print(f'Completed. Results: {args.output_dir}')


if __name__=='__main__':main()
