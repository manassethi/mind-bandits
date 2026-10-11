"""Export per-user click probabilities and category/history policy features."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from src.data.mind import load_news, load_behaviors, features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('models/click_model.joblib'))
    p.add_argument('--data-dir',type=Path,default=Path('dataset/raw/mind-small'))
    p.add_argument('--articles',type=int,default=50)
    p.add_argument('--users',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,default=Path('dataset/processed/q2_environment.npz'))
    args=p.parse_args()
    if args.articles<4 or args.users<1:p.error('Need at least four articles and one user')
    if not args.model.exists():p.error('Train the click model first: python scripts/prepare_simulator.py --full-data --arms 50')
    news=load_news(args.data_dir)
    exposures=Counter()
    for row in load_behaviors(args.data_dir/'train'/'behaviors.tsv'):
        exposures.update(a for a,_ in row['impressions'] if a in news)
    articles=[a for a,_ in exposures.most_common(args.articles)]
    if len(articles)!=args.articles:p.error('Not enough training articles')
    first={}
    for row in load_behaviors(args.data_dir/'dev'/'behaviors.tsv'):
        if row['user'] not in first or row['time']<first[row['user']]['time']:first[row['user']]=row
    rows=list(first.values()); rng=np.random.default_rng(args.seed)
    cohort=[rows[i] for i in rng.choice(len(rows),min(args.users,len(rows)),replace=False)]
    categories=sorted({news[a]['category'] for a in articles})
    feature_names=['bias']+['article_category:'+c for c in categories]+['user_affinity:'+c for c in categories]
    x=np.zeros((len(cohort),len(articles),len(feature_names)))
    probabilities=np.zeros((len(cohort),len(articles)))
    model=joblib.load(args.model)
    for i,row in enumerate(cohort):
        records=[features(row['history'],a,news) for a in articles]
        probabilities[i]=model.predict_proba(records)[:,1]
        for j,(a,record) in enumerate(zip(articles,records)):
            cat=categories.index(news[a]['category'])
            x[i,j,0]=1
            x[i,j,1+cat]=1
            x[i,j,1+len(categories)+cat]=record['category_affinity']
        if (i+1)%200==0:print(f'Prepared {i+1}/{len(cohort)} users...',flush=True)
    # Fixed scale, independent of click labels; features have norm <= 1.
    x/=np.sqrt(3)
    metadata={'articles':[{'article_id':a,**news[a]} for a in articles], 'feature_names':feature_names,
              'category_dimensions':1+len(categories), 'users':len(cohort), 'seed':args.seed,
              'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'description':'Fixed pre-impression histories; category features; oracle probabilities from fitted click model.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,features=x,probabilities=probabilities,metadata=json.dumps(metadata))
    print(f'Saved {args.output}: {len(cohort)} users, {len(articles)} articles, {len(feature_names)} features')


if __name__=='__main__':main()
