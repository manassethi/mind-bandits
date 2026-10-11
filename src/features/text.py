"""Text features for an independent lexical simulator and latent policy features."""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import FeatureHasher
from src.data.mind import features


def article_text(item):
    return item['title'] + ' ' + item.get('abstract', '')


def history_mean(matrix, history, indices):
    rows=[indices[a] for a in history if a in indices]
    if not rows:
        return csr_matrix((1,matrix.shape[1]))
    return csr_matrix(matrix[rows].mean(axis=0))


def reward_matrix(behaviors, news, indices, lexical):
    """Labels stay separate from features; only prior history constructs user context."""
    records, article_rows, labels = [], [], []
    for row in behaviors:
        profile=history_mean(lexical,row['history'],indices)
        norm=float(np.sqrt(profile.multiply(profile).sum()))
        candidates=[(article,label) for article,label in row['impressions'] if article in indices]
        candidate_indices=[indices[a] for a,_ in candidates]
        similarities=(lexical[candidate_indices] @ profile.T).toarray().ravel()/max(norm,1e-12)
        for (article,label),similarity in zip(candidates,similarities):
            index=indices[article]
            record=features(row['history'],article,news)
            # Scale history length; keep numerical inputs on a comparable range.
            record['history_length']/=5
            record['lexical_similarity']=float(similarity)
            record={f'{key}={value}' if isinstance(value,str) else key:1. if isinstance(value,str) else value for key,value in record.items()}
            records.append(record);article_rows.append(index);labels.append(int(label))
    numeric=FeatureHasher(n_features=1024,alternate_sign=False).transform(records)
    return hstack([numeric,lexical[article_rows]],format='csr'),np.asarray(labels)


def text_policy_features(base, article_embeddings, user_embeddings):
    """Append content and user-content interactions; scale extra block only."""
    content=np.broadcast_to(article_embeddings[None,:,:],(len(user_embeddings),)+article_embeddings.shape)
    interaction=content*user_embeddings[:,None,:]
    extra=np.concatenate([content,interaction],axis=2)/np.sqrt(2)
    return np.concatenate([base,extra],axis=2)
