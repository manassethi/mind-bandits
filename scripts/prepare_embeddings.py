"""Add pretrained MiniLM features while preserving the frozen text-aware rewards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from src.data.mind import load_news,load_behaviors,features
from src.features.text import article_text,text_policy_features


def recover_cohort(metadata,train,dev):
    """Replay the source preparation sampling to recover the exact frozen cohort."""
    args=metadata['arguments'];rng=np.random.default_rng(args['seed'])
    if not args['full_data']:
        rng.choice(len(train),min(args['train_impressions'],len(train)),replace=False)
        rng.choice(len(dev),min(args['dev_impressions'],len(dev)),replace=False)
    fit_count=len(train) if args['full_data'] else min(args['train_impressions'],len(train))
    for _ in range(args.get('epochs',3)):
        rng.permutation(fit_count)
    first={}
    for row in dev:
        if row['user'] not in first or row['time']<first[row['user']]['time']:first[row['user']]=row
    rows=list(first.values())
    return [rows[i] for i in rng.choice(len(rows),min(args['users'],len(rows)),replace=False)]


def embedding_feature_view(full_features,history_dimensions,embedding_start):
    return np.concatenate([full_features[:,:,:history_dimensions],full_features[:,:,embedding_start:]],axis=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment',type=Path,default=Path('dataset/processed/q2_text_environment.npz'))
    p.add_argument('--data-dir',type=Path,default=Path('dataset/raw/mind-small'))
    p.add_argument('--model',default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--revision',default=None,help='Optional fixed Hugging Face model commit')
    p.add_argument('--dimensions',type=int,default=16)
    p.add_argument('--projection-articles',type=int,default=5000)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--device',default=None,help='cpu or mps; omitted selects available hardware')
    p.add_argument('--output',type=Path,default=Path('dataset/processed/q2_embeddings_environment.npz'))
    args=p.parse_args()
    if min(args.dimensions,args.projection_articles,args.batch_size)<1:p.error('Sizes must be positive')
    if not args.environment.exists():p.error('Prepare the text simulator first')
    try:
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import model_info
    except ImportError:p.error('Install dependencies: python -m pip install -r requirements-embeddings.txt')
    with np.load(args.environment,allow_pickle=False) as data:
        original=data['features'];probabilities=data['probabilities'];meta=json.loads(str(data['metadata']))
    if 'arguments' not in meta or 'history_dimensions' not in meta:p.error('Need environment from prepare_text_simulator.py')
    if 'embedding_start' in meta:p.error('Use the original text environment, not an already extended one')
    news=load_news(args.data_dir)
    train=list(load_behaviors(args.data_dir/'train'/'behaviors.tsv'))
    dev=list(load_behaviors(args.data_dir/'dev'/'behaviors.tsv'))
    cutoff=min(r['time'] for r in dev);train=[r for r in train if r['time']<cutoff]
    cohort=recover_cohort(meta,train,dev)
    if len(cohort)!=len(original):p.error('Source cohort size mismatch')
    articles=[r['article_id'] for r in meta['articles']]
    categories=sorted({news[a]['category'] for a in articles});history_dim=meta['history_dimensions']
    # Fail closed if raw files or sampling no longer reproduce original user features.
    base=np.zeros_like(original[:,:,:history_dim])
    for i,row in enumerate(cohort):
        for j,a in enumerate(articles):
            k=categories.index(news[a]['category']);record=features(row['history'],a,news)
            base[i,j,0]=1;base[i,j,1+k]=1;base[i,j,1+len(categories)+k]=record['category_affinity']
    base/=np.sqrt(3)
    if not np.allclose(base,original[:,:,:history_dim],atol=1e-12):p.error('Source user histories do not match frozen environment')
    train_news=[]
    with (args.data_dir/'train'/'news.tsv').open() as f:
        for line in f:train_news.append(line.split('\t',1)[0])
    rng=np.random.default_rng(2026)
    projection_ids=[train_news[i] for i in rng.choice(len(train_news),min(args.projection_articles,len(train_news)),replace=False)]
    needed=set(articles)|set(projection_ids)|{a for r in cohort for a in r['history'] if a in news}
    ids=sorted(needed);indices={a:i for i,a in enumerate(ids)}
    texts=[article_text(news[a]) for a in ids]
    revision=args.revision or model_info(args.model).sha
    identity=hashlib.sha256(json.dumps({'model':args.model,'revision':revision,'ids':ids,'texts':texts},sort_keys=True).encode()).hexdigest()
    cache=Path('dataset/processed')/f'embeddings_{identity[:16]}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as data:encoded=data['embeddings']
        print(f'Using cached embeddings for {len(ids):,} articles',flush=True)
    else:
        print(f'Encoding {len(ids):,} required articles with {args.model}...',flush=True)
        model=SentenceTransformer(args.model,revision=revision,device=args.device,trust_remote_code=False)
        encoded=model.encode(texts,batch_size=args.batch_size,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True)
        cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,embeddings=encoded)
    if args.dimensions>=min(len(projection_ids),encoded.shape[1]):p.error('Projection dimensions too large')
    # Fit compression on training news only, never click labels or dev histories.
    projection=PCA(n_components=args.dimensions,random_state=2026,svd_solver='randomized')
    projection.fit(encoded[[indices[a] for a in projection_ids]])
    reduced=normalize(projection.transform(encoded))
    profiles=[]
    for row in cohort:
        history=[indices[a] for a in row['history'] if a in indices]
        profile=reduced[history].mean(axis=0) if history else np.zeros(args.dimensions)
        profiles.append(profile/max(np.linalg.norm(profile),1e-12))
    augmented=text_policy_features(base,reduced[[indices[a] for a in articles]],np.array(profiles))
    extra=augmented[:,:,history_dim:]
    all_features=np.concatenate([original,extra],axis=2)
    meta=dict(meta)
    meta.update({'embedding_start':original.shape[2],'embedding_dimensions':augmented.shape[2],
                 'embedding_model':args.model,'embedding_revision':revision,'embedding_raw_dimensions':encoded.shape[1],
                 'embedding_projection_dimensions':args.dimensions,'embedding_projection_training_articles':len(projection_ids),
                 'embedding_projection_explained_variance':float(projection.explained_variance_ratio_.sum()),
                 'source_environment_sha256':hashlib.sha256(args.environment.read_bytes()).hexdigest(),
                 'embedding_cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),
                 'embedding_description':'Frozen pretrained vectors; train-news PCA; normalized article and mean-history profiles; original rewards unchanged'})
    meta['feature_names']+= [f'pretrained:{i}' for i in range(args.dimensions)]+[f'user_pretrained_interaction:{i}' for i in range(args.dimensions)]
    Path('models').mkdir(exist_ok=True);joblib.dump(projection,'models/embedding_projection.joblib')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,features=all_features,probabilities=probabilities,metadata=json.dumps(meta))
    print(f'Saved {args.output}. Click probabilities and TF-IDF features preserved exactly.')


if __name__=='__main__':main()
