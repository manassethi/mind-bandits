"""Explain a fitted SAE bandit and perturb its input features on fresh contexts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.bandits.contextual import LinUCB
from scripts.run_contextual import available_articles,write_csv


def scores(policy,x):
    mean=x@(policy.inverse@policy.b)
    variance=np.einsum('ij,jk,ik->i',x,policy.inverse,x)
    return mean+policy.scale*np.sqrt(np.maximum(variance,0))


def ablate_features(x,history_dimensions,feature_positions,latent_count):
    result=x.copy()
    for position in feature_positions:
        result[:,history_dimensions+position]=0
        result[:,history_dimensions+latent_count+position]=0
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment',type=Path,default=Path('dataset/processed/q2_sae_environment.npz'))
    p.add_argument('--rounds',type=int,default=10000)
    p.add_argument('--evaluation-contexts',type=int,default=2000)
    p.add_argument('--remove-features',type=int,default=3)
    p.add_argument('--random-controls',type=int,default=10)
    p.add_argument('--alpha',type=float,default=.3)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--pool-seed',type=int,default=2027)
    p.add_argument('--output-dir',type=Path,default=Path('results/sae_interpretability'))
    args=p.parse_args(argv)
    with np.load(args.environment,allow_pickle=False) as data:
        full=data['features'];prob=data['probabilities'];meta=json.loads(str(data['metadata']))
    if 'sae_start' not in meta:p.error('Need an SAE environment')
    h=meta['history_dimensions'];f=len(meta['sae_feature_ids'])
    if not 1<=args.remove_features<=f or min(args.rounds,args.evaluation_contexts,args.random_controls)<1 or args.alpha<0:p.error('Invalid sizes/alpha')
    x=np.concatenate([full[:,:,:h],full[:,:,meta['sae_start']:]],axis=2)
    pool=np.random.default_rng(args.pool_seed).choice(x.shape[1],min(20,x.shape[1]),replace=False)
    candidates=min(5,len(pool)//2);rng=np.random.default_rng(np.random.SeedSequence([args.seed,0]))
    policy=LinUCB(x.shape[2],np.random.default_rng(np.random.SeedSequence([args.seed,1])),args.alpha)
    for t in range(args.rounds):
        user=int(rng.integers(x.shape[0]));arms=rng.choice(available_articles(pool,len(pool)//2,t,args.rounds//2),candidates,replace=False)
        chosen=policy.select(x[user,arms]);reward=int(rng.random()<prob[user,arms[chosen]])
        policy.update(x[user,arms[chosen]],reward)
    # Rank contributions on a separate diagnostic set, before drawing evaluation contexts.
    diagnostic=np.random.default_rng(np.random.SeedSequence([args.seed,2]));weights=policy.inverse@policy.b
    importance=np.zeros(f)
    for _ in range(1000):
        user=int(diagnostic.integers(x.shape[0]));arms=diagnostic.choice(pool,candidates,replace=False)
        z=x[user,arms]
        contribution=z[:,h:h+f]*weights[h:h+f]+z[:,h+f:]*weights[h+f:]
        importance+=np.abs(contribution).mean(axis=0)/1000
    order=np.argsort(-importance);top=order[:args.remove_features]
    evaluation=np.random.default_rng(np.random.SeedSequence([args.seed,3]))
    users=evaluation.integers(x.shape[0],size=args.evaluation_contexts)
    slates=np.array([evaluation.choice(pool,candidates,replace=False) for _ in users])
    samples=x[users[:,None],slates];picks=np.array([np.argmax(scores(policy,z)) for z in samples])
    rewards=prob[users[:,None],slates];best=rewards.max(axis=1)
    baseline_p=rewards[np.arange(len(users)),picks]
    masks=[('baseline',[]) ,('top_features',top.tolist())]
    controls=np.random.default_rng(np.random.SeedSequence([args.seed,4]))
    masks += [(f'random_control_{i+1}',controls.choice(f,args.remove_features,replace=False).tolist()) for i in range(args.random_controls)]
    rows=[]
    for name,positions in masks:
        choices=np.array([np.argmax(scores(policy,ablate_features(z,h,positions,f))) for z in samples])
        selected=rewards[np.arange(len(users)),choices]
        rows.append({'ablation':name,'feature_ids':json.dumps([meta['sae_feature_ids'][i] for i in positions]),
                     'choice_change_fraction':float((choices!=picks).mean()),'expected_ctr':float(selected.mean()),
                     'expected_regret_per_round':float((best-selected).mean()),'expected_ctr_change':float((selected-baseline_p).mean())})
    args.output_dir.mkdir(parents=True,exist_ok=True);write_csv(args.output_dir/'ablations.csv',rows)
    ranked=[{'feature_id':meta['sae_feature_ids'][i],'mean_absolute_predicted_reward_contribution':float(importance[i]),
             'content_weight':float(weights[h+i]),'history_interaction_weight':float(weights[h+f+i])} for i in order]
    write_csv(args.output_dir/'feature_importance.csv',ranked)
    explanations=[]
    for user,arms,z,chosen in zip(users[:20],slates[:20],samples[:20],picks[:20]):
        contributions=z[chosen]*weights
        feature_terms=contributions[h:h+f]+contributions[h+f:]
        explanations.append({'user_cohort_index':int(user),'selected_article':meta['articles'][arms[chosen]],
                             'predicted_linear_reward':float(contributions.sum()),
                             'uncertainty_bonus':float(scores(policy,z)[chosen]-contributions.sum()),
                             'sae_contributions':[{'feature_id':meta['sae_feature_ids'][i],'contribution':float(feature_terms[i])} for i in np.argsort(-np.abs(feature_terms))[:5]]})
    (args.output_dir/'choice_explanations.json').write_text(json.dumps(explanations,indent=2)+'\n')
    (args.output_dir/'config.json').write_text(json.dumps({'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                                                         'environment_sha256':hashlib.sha256(args.environment.read_bytes()).hexdigest(),
                                                         'pool_indices':pool.tolist(),'evaluation':'Frozen policy, no retraining; input perturbations with fixed rewards; not a causal intervention on article semantics'},indent=2)+'\n')
    lines=['# SAE policy sensitivity', '', 'One frozen policy evaluated on fresh contexts. Input removal does not establish causal semantic effects.', '',
           '| Ablation | Choices changed | Expected CTR change | Expected regret per round |', '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['ablation']} | {100*row['choice_change_fraction']:.1f}% | {row['expected_ctr_change']:.5f} | {row['expected_regret_per_round']:.5f} |")
    lines += ['', 'Removed top feature IDs: '+', '.join(str(meta['sae_feature_ids'][i]) for i in top), '',
              'Compare top-feature removal with random controls. Negative CTR changes indicate lower predicted reward. No retraining occurs.']
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Saved feature importance, explanations and ablations to {args.output_dir}')
    for row in rows[:2]:print(row)


if __name__=='__main__':main()
