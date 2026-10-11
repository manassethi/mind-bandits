"""Fit on train impressions; evaluate on dev; export a stationary Q1 environment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from scipy.sparse import vstack
from src.data.mind import load_news, load_behaviors, examples, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('dataset/raw/mind-small'))
    parser.add_argument('--train-impressions', type=int, default=20000)
    parser.add_argument('--dev-impressions', type=int, default=5000)
    parser.add_argument('--cohort-size', type=int, default=1000)
    parser.add_argument('--arms', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--full-data', action='store_true', help='Use all train and dev impressions')
    args = parser.parse_args()
    if min(args.train_impressions, args.dev_impressions, args.cohort_size, args.arms) < 1:
        parser.error('sample sizes and arms must be positive')
    rng = np.random.default_rng(args.seed)
    news = load_news(args.data_dir)
    train = list(load_behaviors(args.data_dir / 'train' / 'behaviors.tsv'))
    dev = list(load_behaviors(args.data_dir / 'dev' / 'behaviors.tsv'))
    # Enforce chronological separation, including any unusual overlap in source files.
    cutoff = min(row['time'] for row in dev)
    train = [row for row in train if row['time'] < cutoff]
    def sample(rows, n):
        return [rows[i] for i in rng.choice(len(rows), min(n, len(rows)), replace=False)]
    selected_train, selected_dev = (train, dev) if args.full_data else (sample(train, args.train_impressions), sample(dev, args.dev_impressions))
    print('Constructing training features...', flush=True)
    # Fit a complete feature vocabulary without labels, then build sparse blocks.
    vectorizer = DictVectorizer()
    vectorizer.fit([features([], article, news) for article in news])
    def matrix(rows):
        blocks, labels = [], []
        for start in range(0, len(rows), 1000):
            x, y = examples(rows[start:start+1000], news)
            blocks.append(vectorizer.transform(x)); labels.append(y)
        return vstack(blocks, format='csr'), np.concatenate(labels)
    x_train, y_train = matrix(selected_train)
    classifier = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    classifier.fit(x_train, y_train)
    del x_train
    model = make_pipeline(vectorizer, classifier)
    print('Evaluating held-out validation impressions...', flush=True)
    x_dev, y_dev = matrix(selected_dev)
    predicted = classifier.predict_proba(x_dev)[:, 1]
    del x_dev
    metrics = {'auc': float(roc_auc_score(y_dev, predicted)), 'log_loss': float(log_loss(y_dev, predicted)),
               'brier_score': float(brier_score_loss(y_dev, predicted)), 'observed_ctr': float(y_dev.mean()),
               'predicted_ctr': float(predicted.mean()), 'train_examples': len(y_train), 'dev_examples': len(y_dev),
               'train_impressions_used': len(selected_train), 'dev_impressions_used': len(selected_dev)}
    # Arm selection depends only on training exposure counts, never dev click labels.
    exposures = Counter(article for row in selected_train for article, _ in row['impressions'] if article in news)
    articles = [article for article, _ in exposures.most_common(args.arms)]
    if len(articles) != args.arms:
        raise ValueError('Not enough articles; increase training impressions')
    # One pre-impression history per unique dev user, sampled independently of labels.
    users = {}
    for row in sorted(dev, key=lambda row: row['time']):
        users.setdefault(row['user'], row['history'])
    cohort = sample(list(users.values()), args.cohort_size)
    arms = []
    for article in articles:
        p = model.predict_proba([features(history, article, news) for history in cohort])[:, 1].mean()
        arms.append({'article_id': article, **news[article], 'click_probability': float(p), 'train_exposures': exposures[article]})
    for folder in ('models', 'dataset/processed', 'results/metrics'):
        Path(folder).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, 'models/click_model.joblib')
    environment = {'seed': args.seed, 'cohort_size': len(cohort), 'description': 'Stationary Bernoulli arms: click-model probabilities averaged over unique dev-user histories.',
                   'model_metrics': metrics, 'arms': arms, 'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}}
    Path('dataset/processed/q1_environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    Path('results/metrics/click_model.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
    print('Environment saved to dataset/processed/q1_environment.json')


if __name__ == '__main__':
    main()
