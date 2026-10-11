"""Repeat frozen-policy feature ablations across article pools and fitted policies."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.analyze_sae import main as analyze
from scripts.run_contextual import write_csv
from src.evaluation.sweep import interval


def run_effects(ablations):
    top=next(r for r in ablations if r['ablation']=='top_features')
    controls=[r for r in ablations if r['ablation'].startswith('random_control_')]
    if not controls:raise ValueError('At least one random control is needed')
    top_drop=-float(top['expected_ctr_change'])
    random_drop=float(np.mean([-float(r['expected_ctr_change']) for r in controls]))
    return {'top_ctr_drop':top_drop,'random_ctr_drop':random_drop,'excess_ctr_drop':top_drop-random_drop,
            'top_choice_change':float(top['choice_change_fraction']),
            'random_choice_change':float(np.mean([float(r['choice_change_fraction']) for r in controls]))}


def effect_summary(runs):
    by_seed=defaultdict(list)
    for row in runs:by_seed[row['seed']].append(row)
    summaries=[]
    for metric in ['top_ctr_drop','random_ctr_drop','excess_ctr_drop','top_choice_change','random_choice_change']:
        values=[np.mean([r[metric] for r in rows]) for rows in by_seed.values()]
        mean,low,high=interval(values)
        summaries.append({'metric':metric,'mean':mean,'ci95_low':low,'ci95_high':high})
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment',type=Path,default=Path('dataset/processed/q2_sae_environment.npz'))
    p.add_argument('--pools',type=int,default=3)
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--rounds',type=int,default=10000)
    p.add_argument('--evaluation-contexts',type=int,default=2000)
    p.add_argument('--remove-features',type=int,default=3)
    p.add_argument('--random-controls',type=int,default=10)
    p.add_argument('--alpha',type=float,default=.3)
    p.add_argument('--seed',type=int,default=2000)
    p.add_argument('--pool-seed',type=int,default=3030)
    p.add_argument('--feature-examples',type=Path,default=Path('results/sae/feature_examples.json'))
    p.add_argument('--output-dir',type=Path,default=Path('results/sae_interpretability_sweep'))
    args=p.parse_args()
    if args.seeds<2 or min(args.pools,args.rounds,args.evaluation_contexts,args.random_controls)<1:p.error('Need >=2 seeds and positive sizes')
    if not args.environment.exists():p.error('Prepare the SAE environment first')
    with np.load(args.environment,allow_pickle=False) as data:meta=json.loads(str(data['metadata']))
    if 'sae_feature_ids' not in meta:p.error('Need SAE feature metadata')
    if not 1<=args.remove_features<=len(meta['sae_feature_ids']):p.error('Invalid removal count')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'environment_sha256':hashlib.sha256(args.environment.read_bytes()).hexdigest(),
            'interval_method':'95% Student t across seed-level pool averages; fixed SAE and user cohort',
            'interpretation':'Frozen-policy input perturbations, no retraining; sensitivity does not prove semantic causality'}
    (args.output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    runs=[];rankings=defaultdict(list)
    for pool in range(args.pools):
        for index in range(args.seeds):
            seed=args.seed+index
            print(f'Interpretability pool {pool+1}/{args.pools}, fitted policy {index+1}/{args.seeds}...',flush=True)
            output=args.output_dir/f'pool_{pool}'/f'seed_{seed}'
            analyze(['--environment',str(args.environment),'--rounds',str(args.rounds),'--evaluation-contexts',str(args.evaluation_contexts),
                     '--remove-features',str(args.remove_features),'--random-controls',str(args.random_controls),
                     '--alpha',str(args.alpha),'--seed',str(seed),'--pool-seed',str(args.pool_seed+pool),'--output-dir',str(output)])
            ablations=list(csv.DictReader((output/'ablations.csv').open()))
            top=next(r for r in ablations if r['ablation']=='top_features')
            runs.append({'pool':pool,'seed':seed,**run_effects(ablations),'top_feature_ids':top['feature_ids']})
            importance=list(csv.DictReader((output/'feature_importance.csv').open()))
            for rank,row in enumerate(importance,1):
                rankings[int(row['feature_id'])].append({'rank':rank,'importance':float(row['mean_absolute_predicted_reward_contribution'])})
            write_csv(args.output_dir/'runs.csv',runs)
    summary=effect_summary(runs);write_csv(args.output_dir/'summary.csv',summary)
    per_pool=[]
    for pool in range(args.pools):
        per_pool += [{'pool':pool,**row} for row in effect_summary([r for r in runs if r['pool']==pool])]
    write_csv(args.output_dir/'per_pool_summary.csv',per_pool)
    stable=[]
    for feature,values in rankings.items():
        stable.append({'feature_id':feature,'top_feature_fraction':float(np.mean([v['rank']<=args.remove_features for v in values])),
                       'mean_rank':float(np.mean([v['rank'] for v in values])),
                       'mean_absolute_reward_contribution':float(np.mean([v['importance'] for v in values]))})
    stable.sort(key=lambda r:(-r['top_feature_fraction'],r['mean_rank']))
    write_csv(args.output_dir/'feature_stability.csv',stable)
    lines=['# Repeated SAE feature ablations','',f'{args.pools} fixed article pools and {args.seeds} independently fitted policies per pool.',
           '', '95% intervals average pools within each seed first. They cover fitted-policy/evaluation randomness for this fixed SAE, cohort and set of pools.', '',
           '| Metric | Mean | 95% interval |','|---|---:|---:|']
    for row in summary:
        lines.append(f"| {row['metric']} | {100*row['mean']:.3f}% | {100*row['ci95_low']:.3f}% to {100*row['ci95_high']:.3f}% |")
    lines += ['', 'CTR drops are absolute percentage points. Positive excess drop means removing top features reduced expected reward more than removing random features.',
              '', 'This is a frozen-policy input sensitivity analysis. It does not show that the units are clean semantic concepts or that article meaning was causally changed.',
              '', 'Feature stability and article examples are in feature_review.md. Full per-policy explanations and random controls are saved in each pool/seed folder.']
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    # Examples are optional; show a warning rather than invent labels if unavailable.
    examples={}
    if args.feature_examples.exists():
        # Verify the example file against the current SAE checkpoint when present.
        checkpoint=Path('models/gpt2_sae.pt')
        if checkpoint.exists() and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==meta['sae_model_sha256']:
            examples={r['feature_id']:r['examples'] for r in json.loads(args.feature_examples.read_text())}
    review=['# SAE feature review','','Numeric IDs identify learned units, not verified concepts. Read these examples before assigning a descriptive label.',
            '', 'Top-feature frequency is the fraction of fitted policies ranking a unit among the selected removal features. It is not a significance test.','']
    if not examples:review += ['Article examples could not be verified against this environment and were omitted.','']
    for row in stable:
        review += [f"## Feature {row['feature_id']}",'',f"Top-feature frequency: {100*row['top_feature_fraction']:.1f}%; mean rank: {row['mean_rank']:.1f}.",'']
        for example in examples.get(row['feature_id'],[])[:5]:
            review.append(f"- {example['title']} ({example['category']}; activation {example['activation']:.3f})")
        review.append('')
    (args.output_dir/'feature_review.md').write_text('\n'.join(review)+'\n')
    for row in summary:print(row)
    print(f'Completed. Reports: {args.output_dir}')


if __name__=='__main__':main()
