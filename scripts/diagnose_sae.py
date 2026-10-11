"""Check SAE redundancy and length/state-norm associations on unseen news text."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
from src.data.mind import load_news
from src.features.text import article_text
from src.features.sae import TopKSAE,preprocess_states
from src.evaluation.sae_diagnostics import correlation,residualize,top_overlap,active_jaccard
from scripts.run_contextual import write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment',type=Path,default=Path('dataset/processed/q2_sae_environment.npz'))
    p.add_argument('--checkpoint',type=Path,default=Path('models/gpt2_sae.pt'))
    p.add_argument('--data-dir',type=Path,default=Path('dataset/raw/mind-small'))
    p.add_argument('--sample-articles',type=int,default=1000)
    p.add_argument('--max-tokens',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--top-examples',type=int,default=10)
    p.add_argument('--seed',type=int,default=4040)
    p.add_argument('--device',default=None)
    p.add_argument('--output-dir',type=Path,default=Path('results/sae_diagnostics'))
    args=p.parse_args()
    if min(args.sample_articles,args.max_tokens,args.batch_size,args.top_examples)<1:p.error('Sizes must be positive')
    if not args.environment.exists() or not args.checkpoint.exists():p.error('Prepare the SAE environment and checkpoint first')
    with np.load(args.environment,allow_pickle=False) as data:
        x=data['features'];meta=json.loads(str(data['metadata']))
    if 'sae_start' not in meta:p.error('Need an SAE environment')
    sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if sha!=meta['sae_model_sha256']:p.error('SAE checkpoint does not match environment; use its original checkpoint or prepare again')
    news=load_news(args.data_dir)
    def catalog(split):
        with (args.data_dir/split/'news.tsv').open() as f:return {line.split('\t',1)[0] for line in f}
    unseen=sorted(catalog('dev')-catalog('train'))
    if len(unseen)<3:p.error('Need at least three dev-only articles for diagnostics')
    rng=np.random.default_rng(args.seed)
    sample=[unseen[i] for i in rng.choice(len(unseen),min(len(unseen),args.sample_articles),replace=False)]
    pool=[a['article_id'] for a in meta['articles']]
    ids=pool+[a for a in sample if a not in pool];texts=[article_text(news[a]) for a in ids]
    tokenizer=AutoTokenizer.from_pretrained(meta['sae_model'],revision=meta['sae_revision'])
    tokenizer.pad_token=tokenizer.eos_token
    lengths=np.array([len(tokens) for tokens in tokenizer(texts,truncation=False,verbose=False)['input_ids']])
    effective=np.minimum(lengths,args.max_tokens)
    device=args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    identity=hashlib.sha256(json.dumps([meta['sae_model'],meta['sae_revision'],meta['sae_layer'],args.max_tokens,ids,texts]).encode()).hexdigest()
    cache=Path('dataset/processed')/f'sae_diagnostic_states_{identity[:16]}.npz'
    if cache.exists():
        with np.load(cache,allow_pickle=False) as data:states=data['states']
        print('Using cached diagnostic GPT-2 states',flush=True)
    else:
        model=AutoModel.from_pretrained(meta['sae_model'],revision=meta['sae_revision'],trust_remote_code=False).to(device).eval()
        model.h=model.h[:meta['sae_layer']];captured={}
        def capture(module,inputs,output):captured['state']=output[0] if isinstance(output,tuple) else output
        hook=model.h[-1].register_forward_hook(capture);batches=[]
        for start in range(0,len(ids),args.batch_size):
            batch=tokenizer(texts[start:start+args.batch_size],padding=True,truncation=True,max_length=args.max_tokens,return_tensors='pt').to(device)
            with torch.no_grad():
                model(**batch,use_cache=False);hidden=captured.pop('state');mask=batch['attention_mask'].unsqueeze(-1)
                batches.append(((hidden*mask).sum(1)/mask.sum(1).clamp_min(1)).cpu().numpy())
            if start//args.batch_size%20==0:print(f'Diagnostic articles: {min(start+args.batch_size,len(ids))}/{len(ids)}',flush=True)
        states=np.concatenate(batches);hook.remove();del model
        if device=='mps':torch.mps.empty_cache()
        cache.parent.mkdir(exist_ok=True);np.savez_compressed(cache,states=states)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if saved.get('max_tokens',args.max_tokens)!=args.max_tokens:p.error('Max tokens must match the checkpoint')
    encoded_states=preprocess_states(states,saved.get('state_normalization','none'))
    sae=TopKSAE(saved['input_dimensions'],saved['latent_dimensions'],saved['top_k'],saved['state_dict']['center'],saved['state_dict']['scale'])
    sae.load_state_dict(saved['state_dict']);sae.eval()
    with torch.no_grad():codes=np.concatenate([sae.encode(torch.tensor(encoded_states[start:start+256])).numpy() for start in range(0,len(states),256)])
    selected=np.array(meta['sae_feature_ids']);raw=codes[:,selected]
    scaled=raw/saved['normalization'];scaled/=np.maximum(np.linalg.norm(scaled,axis=1,keepdims=True),1)
    frozen=x[0,:len(pool),meta['sae_start']:meta['sae_start']+len(selected)]*np.sqrt(2)
    difference=float(np.max(np.abs(scaled[:len(pool)]-frozen)))
    if difference>1e-3:p.error(f'Recomputed features differ from environment by {difference:.4g}; check text, layer and max tokens')
    state_norm=np.linalg.norm(states,axis=1)
    # Primary diagnostics on dev-only catalog; policy pool is a separate stratum.
    strata={'dev_only':np.arange(len(pool),len(ids)),'policy_pool':np.arange(len(pool))}
    feature_rows=[];pair_rows=[];aggregate=[];examples=[]
    for stratum,rows in strata.items():
        controls=np.column_stack([effective[rows],state_norm[rows]])
        residuals=residualize(raw[rows],controls)
        for j,feature in enumerate(selected):
            entry={'stratum':stratum,'feature_id':int(feature),'articles':len(rows),'active_fraction':float((raw[rows,j]>0).mean())}
            for label,values in [('token_length',lengths[rows]),('effective_length',effective[rows]),('state_norm',state_norm[rows])]:
                entry[label+'_pearson']=correlation(raw[rows,j],values)
                entry[label+'_spearman']=correlation(raw[rows,j],values,True)
            feature_rows.append(entry)
        for a in range(len(selected)):
            for b in range(a+1,len(selected)):
                pair_rows.append({'stratum':stratum,'feature_a':int(selected[a]),'feature_b':int(selected[b]),
                                  'pearson':correlation(raw[rows,a],raw[rows,b]),'spearman':correlation(raw[rows,a],raw[rows,b],True),
                                  'scaled_pearson':correlation(scaled[rows,a],scaled[rows,b]),
                                  'length_norm_residual_pearson':correlation(residuals[:,a],residuals[:,b]),
                                  'active_jaccard':active_jaccard(raw[rows,a],raw[rows,b]),
                                  'top_article_overlap':top_overlap(raw[rows,a],raw[rows,b],args.top_examples)})
        aggregate.append({'stratum':stratum,'articles':len(rows),'truncated_fraction':float((lengths[rows]>args.max_tokens).mean()),
                          'total_code_token_length_pearson':correlation(codes[rows].sum(1),lengths[rows]),
                          'total_code_effective_length_pearson':correlation(codes[rows].sum(1),effective[rows]),
                          'total_code_state_norm_pearson':correlation(codes[rows].sum(1),state_norm[rows])})
    for j,feature in enumerate(selected):
        top=np.argsort(-raw[len(pool):,j])[:args.top_examples]+len(pool)
        examples.append({'feature_id':int(feature),'examples':[{'article_id':ids[i],'title':news[ids[i]]['title'],'category':news[ids[i]]['category'],
                                                               'tokens':int(lengths[i]),'activation':float(raw[i,j])} for i in top if raw[i,j]>0]})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.output_dir/'feature_length_correlations.csv',feature_rows);write_csv(args.output_dir/'feature_pairs.csv',pair_rows)
    (args.output_dir/'summary.json').write_text(json.dumps({'strata':aggregate,'source_feature_max_difference':difference},indent=2)+'\n')
    (args.output_dir/'heldout_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    write_csv(args.output_dir/'article_measurements.csv',[{'article_id':a,'stratum':'policy_pool' if i<len(pool) else 'dev_only','tokens':int(lengths[i]),'effective_tokens':int(effective[i]),'state_norm':float(state_norm[i]),'total_code_activation':float(codes[i].sum())} for i,a in enumerate(ids)])
    (args.output_dir/'config.json').write_text(json.dumps({'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'sae_model_sha256':sha,'environment_sha256':hashlib.sha256(args.environment.read_bytes()).hexdigest(),'article_ids':ids},indent=2)+'\n')
    primary=sorted([r for r in pair_rows if r['stratum']=='dev_only' and r['pearson'] is not None],key=lambda r:abs(r['pearson']),reverse=True)
    lines=['# SAE redundancy and length diagnostics','','Primary sample: articles present only in the validation catalog; no click labels are used. Policy-pool results are saved separately.',
           '', 'These are descriptive associations. Residual correlation removes a linear fit to effective token length and GPT-2 state norm; it does not prove a causal explanation.', '',
           '| Feature pair | Raw correlation | After length/norm adjustment | Top-article overlap |','|---|---:|---:|---:|']
    for row in primary[:10]:
        residual='undefined' if row['length_norm_residual_pearson'] is None else f"{row['length_norm_residual_pearson']:.3f}"
        overlap='undefined' if row['top_article_overlap'] is None else f"{100*row['top_article_overlap']:.0f}%"
        lines.append(f"| {row['feature_a']} / {row['feature_b']} | {row['pearson']:.3f} | {residual} | {overlap} |")
    lines += ['', 'Article-length associations for frequently important features:', '']
    stability_path=Path('results/sae_interpretability_sweep/feature_stability.csv')
    stability_config=stability_path.parent/'config.json'
    important=selected[:3].tolist()
    if stability_path.exists() and stability_config.exists():
        import csv
        saved_config=json.loads(stability_config.read_text())
        if saved_config.get('environment_sha256')==hashlib.sha256(args.environment.read_bytes()).hexdigest():
            important=[int(row['feature_id']) for row in list(csv.DictReader(stability_path.open()))[:3]]
    for feature in important:
        found=[r for r in feature_rows if r['stratum']=='dev_only' and r['feature_id']==feature]
        if found:
            row=found[0];lines.append(f"- Feature {feature}: token-length correlation {row['token_length_pearson']}; state-norm correlation {row['state_norm_pearson']}.")
    lines += ['', 'Inspect feature_pairs.csv and feature_length_correlations.csv for all units. Undefined correlations indicate constant/inactive features. High redundancy weakens claims of distinct concepts; length associations can reflect style or content as well as nuisance signals.']
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Saved diagnostics to {args.output_dir}')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
