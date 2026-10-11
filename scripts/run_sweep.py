"""Sweep fixed article pools, horizons and policy settings with paired outcomes."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.bandits.policies import ExploreThenCommit, EpsilonGreedy, UCB1, ThompsonSampling
from src.evaluation.sweep import make_pools, interval
from scripts.run_experiment import simulate


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment', type=Path, default=Path('dataset/processed/q1_environment.json'))
    p.add_argument('--pools', type=int, default=5)
    p.add_argument('--pool-size', type=int, default=5)
    p.add_argument('--rounds', type=int, nargs='+', default=[1000, 10000, 50000])
    p.add_argument('--seeds', type=int, default=30)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--pool-seed', type=int, default=2026)
    p.add_argument('--exploration-fractions', type=float, nargs='+', default=[0.05, 0.1, 0.2])
    p.add_argument('--epsilons', type=float, nargs='+', default=[0.01, 0.05, 0.1])
    p.add_argument('--output-dir', type=Path, default=Path('results/q1_sweep'))
    args = p.parse_args()
    environment = json.loads(args.environment.read_text())
    arms = environment['arms']
    if len(arms) <= args.pool_size and args.pools > 1:
        p.error('Prepare a larger candidate set first: python scripts/prepare_simulator.py --arms 50')
    if args.seeds < 2 or min(args.rounds) <= args.pool_size:
        p.error('Need at least 2 seeds and horizons greater than pool size')
    if not args.exploration_fractions or any(not 0 < x < 1 for x in args.exploration_fractions):
        p.error('Exploration fractions must be strictly between 0 and 1')
    if any(not 0 <= x <= 1 for x in args.epsilons):
        p.error('Epsilons must be in [0, 1]')
    try:
        pools = make_pools(arms, args.pools, args.pool_size, args.pool_seed)
    except ValueError as e:
        p.error(str(e))
    if any(not np.isfinite(a['click_probability']) or not 0 <= a['click_probability'] <= 1 for a in arms):
        p.error('Invalid environment probabilities')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'environment': environment, 'pools': [[arms[i] for i in pool] for pool in pools],
              'interval_method': 'Student t, across independent seeds; aggregate first averages pools within each seed'}
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    runs, summaries, paired = [], [], []
    for pool_id, indices in enumerate(pools):
        probabilities = np.array([arms[i]['click_probability'] for i in indices])
        for horizon in sorted(set(args.rounds)):
            variants = []
            for fraction in sorted(set(args.exploration_fractions)):
                budget = max(args.pool_size, int(horizon * fraction))
                variants.append((f'A/B fraction={fraction:g}', 'A/B', fraction, budget))
            for epsilon in sorted(set(args.epsilons)):
                variants.append((f'Epsilon={epsilon:g}', 'Epsilon greedy', epsilon, 0))
            variants += [('UCB1', 'UCB1', 0., 0), ('Thompson sampling', 'Thompson sampling', 0., 0)]
            grouped = {name: [] for name, *_ in variants}
            print(f'Pool {pool_id+1}/{args.pools}, {horizon:,} rounds, {args.seeds} seeds...', flush=True)
            for seed in range(args.seed, args.seed + args.seeds):
                # Same outcome prefix across horizons and same outcomes across variants.
                uniforms = np.random.default_rng(np.random.SeedSequence([seed, pool_id, 0])).random((horizon, len(indices)))
                seed_rows = {}
                for name, family, setting, budget in variants:
                    family_id = {'A/B': 1, 'Epsilon greedy': 2, 'UCB1': 3, 'Thompson sampling': 4}[family]
                    rng = np.random.default_rng(np.random.SeedSequence([seed, pool_id, family_id]))
                    if family == 'A/B': policy = ExploreThenCommit(len(indices), rng, budget)
                    elif family == 'Epsilon greedy': policy = EpsilonGreedy(len(indices), rng, setting)
                    elif family == 'UCB1': policy = UCB1(len(indices), rng)
                    else: policy = ThompsonSampling(len(indices), rng)
                    clicks, regret = simulate(policy, probabilities, uniforms)
                    row = {'pool': pool_id, 'rounds': horizon, 'seed': seed, 'policy': name,
                           'clicks': int(clicks[-1]), 'ctr': float(clicks[-1]/horizon),
                           'regret': float(regret[-1]), 'regret_per_round': float(regret[-1]/horizon),
                           'exploration_rounds': budget, 'arm_counts': json.dumps(policy.counts.tolist())}
                    runs.append(row); grouped[name].append(row); seed_rows[name] = row
                # Compare each adaptive method against every predeclared A/B budget.
                for baseline, family, *_ in variants:
                    if family != 'A/B': continue
                    for name, other_family, *_ in variants:
                        if other_family == 'A/B': continue
                        paired.append({'pool':pool_id, 'rounds':horizon, 'seed':seed, 'policy':name, 'baseline':baseline,
                                       'click_gain':seed_rows[name]['clicks']-seed_rows[baseline]['clicks'],
                                       'regret_reduction':seed_rows[baseline]['regret']-seed_rows[name]['regret']})
            for name, rows in grouped.items():
                entry = {'pool':pool_id, 'rounds':horizon, 'policy':name}
                for metric in ('clicks','ctr','regret','regret_per_round'):
                    mean, low, high = interval([r[metric] for r in rows])
                    entry.update({metric+'_mean':mean, metric+'_ci95_low':low, metric+'_ci95_high':high})
                summaries.append(entry)
            # Persist completed blocks so a long run has inspectable progress.
            write_csv(args.output_dir/'runs.csv', runs)
            write_csv(args.output_dir/'per_pool_summary.csv', summaries)
    aggregate = []
    for horizon in sorted(set(args.rounds)):
        for name in sorted({r['policy'] for r in runs}):
            entry = {'rounds':horizon, 'policy':name}
            for metric in ('clicks','ctr','regret','regret_per_round'):
                seed_means = [np.mean([r[metric] for r in runs if r['rounds']==horizon and r['policy']==name and r['seed']==seed]) for seed in range(args.seed,args.seed+args.seeds)]
                mean, low, high = interval(seed_means)
                entry.update({metric+'_mean':mean, metric+'_ci95_low':low, metric+'_ci95_high':high})
            aggregate.append(entry)
    comparisons=[]
    for horizon, name, baseline in sorted({(r['rounds'],r['policy'],r['baseline']) for r in paired}):
        entry={'rounds':horizon,'policy':name,'baseline':baseline}
        for metric in ('click_gain','regret_reduction'):
            values=[np.mean([r[metric] for r in paired if (r['rounds'],r['policy'],r['baseline'],r['seed'])==(horizon,name,baseline,seed)]) for seed in range(args.seed,args.seed+args.seeds)]
            mean,low,high=interval(values)
            entry.update({metric+'_mean':mean,metric+'_ci95_low':low,metric+'_ci95_high':high})
        comparisons.append(entry)
    write_csv(args.output_dir/'summary.csv', aggregate)
    write_csv(args.output_dir/'paired_comparisons.csv', comparisons)
    fig, axes=plt.subplots(1,2,figsize=(13,5))
    for name in sorted({r['policy'] for r in aggregate}):
        rows=sorted([r for r in aggregate if r['policy']==name],key=lambda r:r['rounds'])
        for ax,metric in zip(axes,('ctr','regret_per_round')):
            xs=[r['rounds'] for r in rows]; means=np.array([r[metric+'_mean'] for r in rows])
            low=np.array([r[metric+'_ci95_low'] for r in rows]); high=np.array([r[metric+'_ci95_high'] for r in rows])
            ax.errorbar(xs,means,yerr=[means-low,high-means],marker='o',capsize=3,label=name)
            ax.set_xscale('log'); ax.grid(alpha=.2); ax.set_xlabel('Rounds (log scale)')
    axes[0].set_ylabel('Mean click rate'); axes[1].set_ylabel('Mean expected regret per round')
    axes[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1))
    fig.suptitle(f'Q1 sweep: average across {args.pools} fixed pools; 95% t intervals across {args.seeds} seeds')
    fig.tight_layout(); fig.savefig(args.output_dir/'comparison.png',dpi=180); plt.close(fig)
    print(f'Completed. Results: {args.output_dir}')


if __name__ == '__main__': main()
