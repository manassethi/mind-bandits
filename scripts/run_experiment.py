"""Compare noncontextual policies in the fitted stationary MIND simulator."""
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


def simulate(policy, probabilities, uniforms):
    clicks, regret = np.zeros(len(uniforms)), np.zeros(len(uniforms))
    for t in range(len(uniforms)):
        arm = policy.select()
        reward = int(uniforms[t, arm] < probabilities[arm])
        policy.update(arm, reward)
        clicks[t] = reward
        regret[t] = probabilities.max() - probabilities[arm]
    return clicks.cumsum(), regret.cumsum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=Path, default=Path('dataset/processed/q1_environment.json'))
    parser.add_argument('--rounds', type=int, default=10000)
    parser.add_argument('--seeds', type=int, default=30)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--exploration-rounds', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--output-dir', type=Path, default=Path('results/q1'))
    args = parser.parse_args()
    environment = json.loads(args.environment.read_text())
    probabilities = np.array([arm['click_probability'] for arm in environment['arms']])
    if args.rounds < 1 or args.seeds < 2 or len(probabilities) < 2:
        parser.error('need positive rounds, at least two seeds and two arms')
    if not len(probabilities) <= args.exploration_rounds < args.rounds:
        parser.error('exploration rounds must cover every arm and be less than total rounds')
    if not 0 <= args.epsilon <= 1 or not np.all((probabilities >= 0) & (probabilities <= 1)):
        parser.error('epsilon and probabilities must be in [0, 1]')
    factories = {
        'A/B (explore then commit)': lambda rng: ExploreThenCommit(len(probabilities), rng, args.exploration_rounds),
        'Epsilon greedy': lambda rng: EpsilonGreedy(len(probabilities), rng, args.epsilon),
        'UCB1': lambda rng: UCB1(len(probabilities), rng),
        'Thompson sampling': lambda rng: ThompsonSampling(len(probabilities), rng),
    }
    trajectories = {name: [] for name in factories}
    rows = []
    for seed in range(args.seed, args.seed + args.seeds):
        # Shared potential outcomes for each round/arm; policies only receive chosen rewards.
        uniforms = np.random.default_rng(np.random.SeedSequence([seed, 0])).random((args.rounds, len(probabilities)))
        for index, (name, factory) in enumerate(factories.items(), 1):
            policy = factory(np.random.default_rng(np.random.SeedSequence([seed, index])))
            clicks, regret = simulate(policy, probabilities, uniforms)
            trajectories[name].append((clicks, regret))
            rows.append({'policy': name, 'seed': seed, 'clicks': int(clicks[-1]), 'ctr': float(clicks[-1] / args.rounds),
                         'cumulative_regret': float(regret[-1]), 'arm_counts': json.dumps(policy.counts.tolist())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries = []
    for name, runs in trajectories.items():
        values = np.array(runs)
        entry = {'policy': name}
        for label, index in [('clicks', 0), ('cumulative_regret', 1)]:
            finals = values[:, index, -1]
            entry[label + '_mean'] = float(finals.mean())
            entry[label + '_std'] = float(finals.std(ddof=1))
            entry[label + '_ci95_half_width'] = float(1.96 * finals.std(ddof=1) / np.sqrt(args.seeds))
        summaries.append(entry)
    with (args.output_dir / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    (args.output_dir / 'config.json').write_text(json.dumps({'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                                                         'environment': environment}, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    t = np.arange(1, args.rounds + 1)
    for name, runs in trajectories.items():
        values = np.array(runs)
        for index, ax in enumerate(axes):
            mean = values[:, index].mean(axis=0)
            ci = 1.96 * values[:, index].std(axis=0, ddof=1) / np.sqrt(args.seeds)
            ax.plot(t, mean, label=name)
            ax.fill_between(t, mean-ci, mean+ci, alpha=0.15)
    axes[0].plot(t, t * probabilities.max(), '--', color='black', label='Oracle expected clicks')
    for ax, label in zip(axes, ['Cumulative clicks', 'Cumulative expected regret']):
        ax.set(xlabel='Recommendation round', ylabel=label)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'MIND-small stationary simulator | {args.seeds} runs | shaded: approximate 95% CI')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'comparison.png', dpi=180)
    plt.close(fig)
    for row in summaries:
        print(f"{row['policy']:28s} clicks={row['clicks_mean']:.1f} regret={row['cumulative_regret_mean']:.1f}")
    print(f'Outputs saved to {args.output_dir}')


if __name__ == '__main__':
    main()
