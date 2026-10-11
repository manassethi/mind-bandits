"""Train an exploratory SAE on GPT-2 states; preserve the existing reward model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from transformers import AutoModel,AutoTokenizer
from huggingface_hub import model_info
from src.data.mind import load_news,load_behaviors
from src.features.text import article_text,text_policy_features
from src.features.sae import TopKSAE,preprocess_states
from scripts.prepare_embeddings import recover_cohort


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment',type=Path,default=Path('dataset/processed/q2_embeddings_environment.npz'))
    p.add_argument('--data-dir',type=Path,default=Path('dataset/raw/mind-small'))
    p.add_argument('--model',default='openai-community/gpt2')
    p.add_argument('--revision',default=None)
    p.add_argument('--layer',type=int,default=8)
    p.add_argument('--train-articles',type=int,default=5000)
    p.add_argument('--latent-dimensions',type=int,default=1024)
    p.add_argument('--top-k',type=int,default=16)
    p.add_argument('--policy-features',type=int,default=32)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--max-tokens',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default=None)
    p.add_argument('--cohort-limit',type=int,default=None,help='Optional first-N source users for a lighter pipeline check')
    p.add_argument('--output',type=Path,default=Path('dataset/processed/q2_sae_environment.npz'))
    p.add_argument('--state-normalization',choices=['none','unit'],default='none',help='Normalize each pooled state before SAE training and matched dense PCA')
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--report-dir',type=Path,default=None)
    args=p.parse_args()
    suffix='_normalized' if args.state_normalization=='unit' else ''
    if args.output==Path('dataset/processed/q2_sae_environment.npz') and suffix:
        args.output=Path(f'dataset/processed/q2_sae{suffix}_environment.npz')
    args.checkpoint=args.checkpoint or Path(f'models/gpt2_sae{suffix}.pt')
    args.report_dir=args.report_dir or Path(f'results/sae{suffix}')
    if min(args.train_articles,args.latent_dimensions,args.top_k,args.policy_features,args.epochs,args.batch_size,args.max_tokens)<1 or args.train_articles<20 or args.policy_features>args.latent_dimensions or args.top_k>args.latent_dimensions:
        p.error('Invalid sizes; need >=20 training articles and valid feature/top-k counts')
    if not args.environment.exists():p.error('Run prepare_embeddings.py first')
    with np.load(args.environment,allow_pickle=False) as data:
        original=data['features'];prob=data['probabilities'];meta=json.loads(str(data['metadata']))
    if 'embedding_start' not in meta or 'sae_start' in meta:p.error('Use an unextended pretrained embedding environment')
    device=args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    news=load_news(args.data_dir)
    train=list(load_behaviors(args.data_dir/'train'/'behaviors.tsv'));dev=list(load_behaviors(args.data_dir/'dev'/'behaviors.tsv'))
    cutoff=min(r['time'] for r in dev);train=[r for r in train if r['time']<cutoff]
    cohort=recover_cohort(meta,train,dev)
    if args.cohort_limit is not None:
        if args.cohort_limit<1:p.error('Cohort limit must be positive')
        cohort=cohort[:args.cohort_limit];original=original[:len(cohort)];prob=prob[:len(cohort)]
        meta=dict(meta);meta['users']=len(cohort)
    with (args.data_dir/'train'/'news.tsv').open() as f:train_ids=[line.split('\t',1)[0] for line in f]
    sampled=[train_ids[i] for i in rng.choice(len(train_ids),min(args.train_articles,len(train_ids)),replace=False)]
    split=max(1,int(.9*len(sampled)));fit_ids=sampled[:split];valid_ids=sampled[split:]
    articles=[r['article_id'] for r in meta['articles']]
    needed=set(sampled)|set(articles)|{a for row in cohort for a in row['history'] if a in news}
    ids=sorted(needed);index={a:i for i,a in enumerate(ids)};texts=[article_text(news[a]) for a in ids]
    revision=args.revision or model_info(args.model).sha
    identity=hashlib.sha256(json.dumps([args.model,revision,args.layer,args.max_tokens,ids,texts]).encode()).hexdigest()
    cache=Path('dataset/processed')/f'gpt2_states_{identity[:16]}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as data:states=data['states']
        print(f'Using cached GPT-2 states for {len(ids):,} articles',flush=True)
    else:
        tokenizer=AutoTokenizer.from_pretrained(args.model,revision=revision)
        tokenizer.pad_token=tokenizer.eos_token
        model=AutoModel.from_pretrained(args.model,revision=revision,trust_remote_code=False).to(device).eval()
        if not 1<=args.layer<=model.config.n_layer:p.error('Layer out of range')
        # Stop after the requested block and capture its raw residual state.
        model.h=model.h[:args.layer]
        captured={}
        def capture(module,inputs,output):
            captured['state']=output[0] if isinstance(output,tuple) else output
        hook=model.h[-1].register_forward_hook(capture)
        batches=[]
        for start in range(0,len(ids),args.batch_size):
            tokens=tokenizer(texts[start:start+args.batch_size],padding=True,truncation=True,max_length=args.max_tokens,return_tensors='pt').to(device)
            with torch.no_grad():
                model(**tokens,output_hidden_states=False,use_cache=False)
                hidden=captured.pop('state')
                mask=tokens['attention_mask'].unsqueeze(-1)
                pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
                batches.append(pooled.cpu().numpy())
            if start//args.batch_size%50==0:print(f'GPT-2 states: {min(start+args.batch_size,len(ids)):,}/{len(ids):,} articles',flush=True)
        states=np.concatenate(batches)
        np.savez_compressed(cache,states=states)
        hook.remove();del model
        if device=='mps':torch.mps.empty_cache()
    states=preprocess_states(states,args.state_normalization)
    fit=states[[index[a] for a in fit_ids]];valid=states[[index[a] for a in valid_ids]]
    if args.latent_dimensions<=fit.shape[1]:p.error('Latent dimensions must exceed GPT-2 input dimensions for an overcomplete dictionary')
    if args.policy_features>=min(fit.shape):p.error('Too many policy features for dense GPT-2 control')
    center=fit.mean(axis=0);scale=max(float(np.sqrt(np.mean((fit-center)**2))),1e-6)
    sae=TopKSAE(fit.shape[1],args.latent_dimensions,args.top_k,center,scale).to(device)
    optimizer=torch.optim.Adam(sae.parameters(),lr=1e-3)
    train_tensor=torch.tensor(fit,device=device);val_tensor=torch.tensor(valid,device=device)
    curve=[]
    for epoch in range(args.epochs):
        losses=[]
        for indices in np.array_split(rng.permutation(len(fit)),max(1,int(np.ceil(len(fit)/256)))):
            batch=train_tensor[torch.tensor(indices,device=device)]
            prediction,_=sae(batch);loss=((prediction-batch)/scale).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step();sae.normalize_decoder();losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            prediction,_=sae(val_tensor);val_loss=float(((prediction-val_tensor)/scale).square().mean().cpu())
        curve.append({'epoch':epoch+1,'train_normalized_mse':float(np.mean(losses)),'validation_normalized_mse':val_loss})
        print(f'SAE epoch {epoch+1}/{args.epochs}: held-out normalized MSE={val_loss:.4f}',flush=True)
    sae.eval();code_batches=[]
    with torch.no_grad():
        for start in range(0,len(states),256):code_batches.append(sae.encode(torch.tensor(states[start:start+256],device=device)).cpu().numpy())
    codes=np.concatenate(code_batches);fit_codes=codes[[index[a] for a in fit_ids]];val_codes=codes[[index[a] for a in valid_ids]]
    # Select only from SAE-training activations, never bandit outcomes or oracle rewards.
    selected=np.argsort(-fit_codes.mean(axis=0))[:args.policy_features]
    retained=codes[:,selected]
    normalization=max(float(np.quantile(np.linalg.norm(fit_codes[:,selected],axis=1),.95)),1e-8)
    retained/=normalization
    retained/=np.maximum(np.linalg.norm(retained,axis=1,keepdims=True),1)
    profiles=[]
    for row in cohort:
        history=[index[a] for a in row['history'] if a in index]
        profile=retained[history].mean(axis=0) if history else np.zeros(args.policy_features)
        profiles.append(profile/max(np.linalg.norm(profile),1e-12))
    base=original[:,:,:meta['history_dimensions']]
    augmented=text_policy_features(base,retained[[index[a] for a in articles]],np.array(profiles))
    # Matched-width dense GPT-2 control separates source-model effects from SAE features.
    projection=PCA(n_components=args.policy_features,random_state=args.seed,svd_solver='randomized')
    projection.fit(fit)
    dense=normalize(projection.transform(states))
    dense_profiles=[]
    for row in cohort:
        history=[index[a] for a in row['history'] if a in index]
        profile=dense[history].mean(axis=0) if history else np.zeros(args.policy_features)
        dense_profiles.append(profile/max(np.linalg.norm(profile),1e-12))
    dense_augmented=text_policy_features(base,dense[[index[a] for a in articles]],np.array(dense_profiles))
    dense_extra=dense_augmented[:,:,meta['history_dimensions']:]
    sae_start=original.shape[2]+dense_extra.shape[2]
    x=np.concatenate([original,dense_extra,augmented[:,:,meta['history_dimensions']:]],axis=2)
    metrics={'heldout_normalized_mse':curve[-1]['validation_normalized_mse'],
             'heldout_mean_baseline_normalized_mse':float(np.mean(((valid-center)/scale)**2)),
             'heldout_mean_nonzero_codes':float((val_codes>0).sum(axis=1).mean()),
             'training_active_dictionary_fraction':float((fit_codes.max(axis=0)>0).mean()),
             'selected_training_activation_mass_fraction':float(fit_codes[:,selected].sum()/max(fit_codes.sum(),1e-8)),
             'fit_articles':len(fit_ids),'heldout_articles':len(valid_ids),'training_curve':curve}
    examples=[]
    for feature in selected:
        top=np.argsort(-codes[:,feature])[:8]
        examples.append({'feature_id':int(feature),'training_activation_frequency':float((fit_codes[:,feature]>0).mean()),
                         'examples':[{'article_id':ids[i],'title':news[ids[i]]['title'],'category':news[ids[i]]['category'],'activation':float(codes[i,feature])} for i in top if codes[i,feature]>0]})
    model_path=args.checkpoint;model_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':{k:v.cpu() for k,v in sae.state_dict().items()},'input_dimensions':fit.shape[1],'latent_dimensions':args.latent_dimensions,'top_k':args.top_k,'selected_features':selected.tolist(),'normalization':normalization,'gpt2_revision':revision,'layer':args.layer,'state_normalization':args.state_normalization,'max_tokens':args.max_tokens},model_path)
    meta=dict(meta);meta.update({'embedding_end':original.shape[2],'gpt2_start':original.shape[2],'gpt2_end':sae_start,'gpt2_dimensions':dense_augmented.shape[2],'sae_start':sae_start,'sae_dimensions':augmented.shape[2],
                                'sae_feature_ids':selected.tolist(),'sae_model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
                                'sae_state_normalization':args.state_normalization,'sae_max_tokens':args.max_tokens,'sae_checkpoint':str(model_path),'sae_metrics':metrics,'sae_model':args.model,'sae_revision':revision,'sae_layer':args.layer,'sae_top_k':args.top_k,
                                'sae_latent_dimensions':args.latent_dimensions,'source_environment_sha256':hashlib.sha256(args.environment.read_bytes()).hexdigest(),
                                'sae_description':'Locally trained top-k SAE on mean-pooled GPT-2 layer states; feature subset chosen on training activations only'})
    meta['feature_names'] += [f'gpt2_pca:{i}' for i in range(args.policy_features)]+[f'user_gpt2_interaction:{i}' for i in range(args.policy_features)]
    meta['feature_names'] += [f'sae:{f}' for f in selected]+[f'user_sae_interaction:{f}' for f in selected]
    args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,features=x,probabilities=prob,metadata=json.dumps(meta))
    report=args.report_dir;report.mkdir(parents=True,exist_ok=True)
    (report/'feature_examples.json').write_text(json.dumps(examples,indent=2)+'\n');(report/'reconstruction.json').write_text(json.dumps(metrics,indent=2)+'\n')
    lines=['# SAE feature examples', '', 'These are high-activation articles, not verified semantic concept labels.', '']
    for entry in examples:
        lines += [f"## Feature {entry['feature_id']}", '']
        for example in entry['examples'][:5]:
            lines.append(f"- {example['title']} ({example['category']}; activation {example['activation']:.3f})")
        lines.append('')
    (report/'feature_examples.md').write_text('\n'.join(lines)+'\n')
    print(f'Saved {args.output} and SAE feature examples in {report}')


if __name__=='__main__':main()
