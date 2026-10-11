"""Read MIND TSVs and construct features from pre-impression history only."""
import csv
from collections import Counter
from datetime import datetime
import numpy as np


def load_news(root):
    news = {}
    for split in ('train', 'dev'):
        with (root / split / 'news.tsv').open(encoding='utf-8') as f:
            for row in csv.reader(f, delimiter='\t'):
                news[row[0]] = {'category': row[1], 'subcategory': row[2], 'title': row[3], 'abstract': row[4]}
    return news


def load_behaviors(path):
    with path.open(encoding='utf-8') as f:
        for row in csv.reader(f, delimiter='\t'):
            yield {'user': row[1], 'time': datetime.strptime(row[2], '%m/%d/%Y %I:%M:%S %p'),
                   'history': row[3].split(), 'impressions': [token.rsplit('-', 1) for token in row[4].split()]}


def features(history, article, news):
    item = news[article]
    categories = Counter(news[n]['category'] for n in history if n in news)
    subcategories = Counter(news[n]['subcategory'] for n in history if n in news)
    n = max(sum(categories.values()), 1)
    affinity = categories[item['category']] / n
    return {'category': item['category'], 'subcategory': item['subcategory'],
            'history_length': float(np.log1p(len(history))), 'category_affinity': affinity,
            'subcategory_affinity': subcategories[item['subcategory']] / n,
            'seen_before': float(article in history),
            'affinity:' + item['category']: affinity}


def examples(behaviors, news):
    x, y = [], []
    for row in behaviors:
        for article, label in row['impressions']:
            if article in news:
                x.append(features(row['history'], article, news))
                y.append(int(label))
    return x, np.asarray(y)
