"""Train a lexical click model and export category/history/latent-text features."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss
from src.data.mind import load_news,load_behaviors,features
from src.features.text import article_text,reward_matrix,text_policy_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('dataset/raw/mind-small'))
    p.add_argument('--full-data',action='store_true')
    p.add_argument('--train-impressions',type=int,default=20000)
    p.add_argument('--dev-impressions',type=int,default=5000)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--articles',type=int,default=50)
    p.add_argument('--users',type=int,default=1000)
    p.add_argument('--dimensions',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,default=Path('dataset/processed/q2_text_environment.npz'))
    args=p.parse_args()
    if min(args.train_impressions,args.dev_impressions,args.epochs,args.batch_size,args.users,args.dimensions)<1 or args.articles<4:
        p.error('Sizes must be positive; need at least four articles')
    news=load_news(args.data_dir);ids=list(news);indices={a:i for i,a in enumerate(ids)}
    texts=[article_text(news[a]) for a in ids]
    train=list(load_behaviors(args.data_dir/'train'/'behaviors.tsv'))
    dev=list(load_behaviors(args.data_dir/'dev'/'behaviors.tsv'))
    cutoff=min(row['time'] for row in dev)
    train=[r for r in train if r['time']<cutoff]
    rng=np.random.default_rng(args.seed)
    def sample(rows,n):return [rows[i] for i in rng.choice(len(rows),min(n,len(rows)),replace=False)]
    fit_rows=train if args.full_data else sample(train,args.train_impressions)
    evaluation=dev if args.full_data else sample(dev,args.dev_impressions)
    hasher=HashingVectorizer(n_features=2**16,alternate_sign=False,stop_words='english',norm='l2')
    lexical=hasher.transform(texts)
    classifier=SGDClassifier(loss='log_loss',alpha=1e-5,learning_rate='constant',eta0=0.01,average=True,random_state=args.seed)
    # Unweighted binary labels preserve click prevalence rather than balancing classes.
    for epoch in range(args.epochs):
        order=rng.permutation(len(fit_rows)); seen=0
        for start in range(0,len(order),args.batch_size):
            batch=[fit_rows[i] for i in order[start:start+args.batch_size]]
            matrix,labels=reward_matrix(batch,news,indices,lexical)
            classifier.partial_fit(matrix,labels,classes=np.array([0,1]));seen+=len(labels)
            if start//args.batch_size%50==0:print(f'Epoch {epoch+1}/{args.epochs}: {start+len(batch):,}/{len(order):,} impressions',flush=True)
        print(f'Epoch {epoch+1} finished: {seen:,} labeled candidates',flush=True)
    truth=[];predictions=[]
    for start in range(0,len(evaluation),args.batch_size):
        matrix,labels=reward_matrix(evaluation[start:start+args.batch_size],news,indices,lexical)
        truth.append(labels);predictions.append(classifier.predict_proba(matrix)[:,1])
    y=np.concatenate(truth);pred=np.concatenate(predictions)
    metrics={'auc':float(roc_auc_score(y,pred)),'log_loss':float(log_loss(y,pred)),'brier_score':float(brier_score_loss(y,pred)),
             'observed_ctr':float(y.mean()),'predicted_ctr':float(pred.mean()),'train_impressions_used':len(fit_rows),
             'dev_impressions_used':len(evaluation),'train_examples':seen,'dev_examples':len(y)}
    print(json.dumps(metrics,indent=2),flush=True)
    # Unsupervised policy representation: fit vocabulary/SVD on training catalog only.
    train_ids=set()
    with (args.data_dir/'train'/'news.tsv').open() as f:
        for line in f:train_ids.add(line.split('\t',1)[0])
    train_indices=[indices[a] for a in ids if a in train_ids]
    tfidf=TfidfVectorizer(max_features=20000,min_df=2,stop_words='english',sublinear_tf=True)
    training_text=tfidf.fit_transform([texts[i] for i in train_indices])
    if args.dimensions>=min(training_text.shape):p.error('Too many latent dimensions for text matrix')
    print('Fitting TF-IDF + SVD policy representation...',flush=True)
    svd=TruncatedSVD(n_components=args.dimensions,random_state=args.seed)
    svd.fit(training_text)
    embeddings=normalize(svd.transform(tfidf.transform(texts)))
    exposures=Counter(a for row in fit_rows for a,_ in row['impressions'] if a in news)
    articles=[a for a,_ in exposures.most_common(args.articles)]
    if len(articles)<args.articles:p.error('Not enough articles in training sample')
    first={}
    for row in dev:
        if row['user'] not in first or row['time']<first[row['user']]['time']:first[row['user']]=row
    cohort=sample(list(first.values()),args.users)
    categories=sorted({news[a]['category'] for a in articles});cat_index={c:i for i,c in enumerate(categories)}
    history_dim=1+2*len(categories)
    base=np.zeros((len(cohort),len(articles),history_dim));probabilities=np.zeros(base.shape[:2]);profiles=[]
    for i,row in enumerate(cohort):
        history_indices=[indices[a] for a in row['history'] if a in indices]
        profile=embeddings[history_indices].mean(axis=0) if history_indices else np.zeros(args.dimensions)
        profiles.append(profile/max(np.linalg.norm(profile),1e-12))
        fake={'history':row['history'],'impressions':[(a,'0') for a in articles]}
        matrix,_=reward_matrix([fake],news,indices,lexical)
        probabilities[i]=classifier.predict_proba(matrix)[:,1]
        for j,a in enumerate(articles):
            k=cat_index[news[a]['category']];record=features(row['history'],a,news)
            base[i,j,0]=1;base[i,j,1+k]=1;base[i,j,1+len(categories)+k]=record['category_affinity']
    base/=np.sqrt(3)
    x=text_policy_features(base,embeddings[[indices[a] for a in articles]],np.array(profiles))
    model_path=Path('models/text_click_model.joblib');model_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'classifier':classifier,'hasher':hasher,'numeric_hash_dimensions':1024},model_path)
    joblib.dump({'tfidf':tfidf,'svd':svd},'models/text_representation.joblib')
    meta={'articles':[{'article_id':a,**news[a]} for a in articles], 'users':len(cohort),'seed':args.seed,
          'category_dimensions':1+len(categories),'history_dimensions':history_dim,'text_dimensions':x.shape[2],
          'feature_names':['bias']+['category:'+c for c in categories]+['affinity:'+c for c in categories]+[f'text:{i}' for i in range(args.dimensions)]+[f'user_text_interaction:{i}' for i in range(args.dimensions)],
          'representation':'TF-IDF + truncated SVD, fitted on training news text without labels; not neural embeddings',
          'reward_model':'SGD logistic: hashed title/abstract terms, category/history and lexical history cosine similarity',
          'model_metrics':metrics,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
          'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,features=x,probabilities=probabilities,metadata=json.dumps(meta))
    Path('results/metrics').mkdir(parents=True,exist_ok=True)
    Path('results/metrics/text_click_model.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(f'Saved {args.output}: {len(cohort)} users, {len(articles)} articles, {x.shape[2]} features')


if __name__=='__main__':main()
