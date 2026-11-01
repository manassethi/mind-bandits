"""Extract policy-side article representations (never the simulator's encoder).

Gemma 3 models: mean-pooled residual stream after every decoder layer (BOS and
padding excluded), plus Gemma Scope 2 JumpReLU SAE codes at the labeled layers,
encoded per token and mean-pooled. Sentence encoder: normalized embeddings.

Outputs go to dataset/processed/policy_raw/<name>/ (large; kept on the cluster):
    dense_layer_<L>.npy   (N, d_model) float32 (Gemma has "massive activation" dims near
                          65k, past float16's range)
    sae_layer_<L>.npz     scipy CSR (N, 16384) float32 values
    meta.json             token stats, SAE L0 and reconstruction quality

    python -m scripts.extract_policy_features --model gemma-3-1b
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from src.data.mind import load_processed_news
from src.simulator.encode import article_text

RAW_DIR = Path("dataset/processed/policy_raw")
GEMMA = {  # name -> (HF model, Gemma Scope 2 repo, labeled SAE layers)
    "gemma-3-1b": ("google/gemma-3-1b-pt", "google/gemma-scope-2-1b-pt", (7, 13, 17, 22)),
    "gemma-3-12b": ("google/gemma-3-12b-pt", "google/gemma-scope-2-12b-pt", (12, 24, 31, 41)),
}
SAE_VARIANT = "width_16k_l0_medium"
SAE_CHUNK = 8192  # tokens encoded at once
SENTENCE = {"bge-m3": "BAAI/bge-m3"}  # 8192-token context: no MIND article is truncated


class _Stop(Exception):
    pass


class JumpReLUSAE:
    def __init__(self, repo: str, layer: int, device):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(repo, f"resid_post/layer_{layer}_{SAE_VARIANT}/params.safetensors")
        p = {k.lower(): v.to(device, torch.float32) for k, v in load_file(path).items()}
        self.w_enc, self.b_enc, self.threshold = p["w_enc"], p["b_enc"], p["threshold"]
        self.w_dec, self.b_dec = p["w_dec"], p["b_dec"]
        self.width = self.w_enc.shape[1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.w_dec + self.b_dec


def decoder_layers(model, n_layers: int) -> torch.nn.ModuleList:
    for name, mod in model.named_modules():
        if (isinstance(mod, torch.nn.ModuleList) and name.endswith("layers")
                and "vision" not in name and len(mod) == n_layers):
            return mod
    raise RuntimeError("text decoder layers not found")


def load_gemma(model_id: str, device):
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(model_id)
    n_layers = getattr(cfg, "text_config", cfg).num_hidden_layers
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            model = getattr(transformers, cls).from_pretrained(model_id, dtype=torch.bfloat16).to(device)
            break
        except (ValueError, KeyError):
            continue
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    return model.eval(), tok, n_layers


def batches_by_tokens(lengths: np.ndarray, max_tokens: int):
    order = np.argsort(-lengths, kind="stable")
    i = 0
    while i < len(order):
        n = max(1, max_tokens // int(lengths[order[i]]))
        yield order[i:i + n]
        i += n


@torch.no_grad()
def extract_gemma(name: str, texts: list[str], out: Path, max_tokens: int, device) -> dict:
    model_id, sae_repo, sae_layers = GEMMA[name]
    model, tok, n_layers = load_gemma(model_id, device)
    saes = {L: JumpReLUSAE(sae_repo, L, device) for L in sae_layers}
    lengths = np.array([len(ids) for ids in tok(texts)["input_ids"]])
    assert tok(texts[:1])["input_ids"][0][0] == tok.bos_token_id
    d_model = model.config.get_text_config().hidden_size
    N = len(texts)

    # Held in RAM and written once per layer: batches arrive in length order, and scattered
    # row writes to a memmap on the cluster's network filesystem stall the GPU.
    dense = {L: np.empty((N, d_model), np.float32) for L in range(n_layers)}
    sae_rows: dict[int, list] = {L: [] for L in sae_layers}
    sae_stats = {L: {"tokens": 0, "l0_sum": 0.0, "sq_err": 0.0, "sq_tot": 0.0} for L in sae_layers}
    state: dict = {}

    def hook(L):
        def fn(_mod, _inp, output):
            h = (output[0] if isinstance(output, tuple) else output).float()
            mask = state["mask"]
            denom = mask.sum(1, keepdim=True)
            dense[L][state["idx"]] = ((h * mask[..., None]).sum(1) / denom).cpu().numpy()
            if L in saes:
                x = h[mask.bool()]
                tok_article = torch.repeat_interleave(torch.arange(len(mask), device=device), mask.sum(1).long())
                pooled = torch.zeros(len(mask), saes[L].width, device=device)
                st = sae_stats[L]
                st["sq_tot"] += float(((x - x.mean(0)) ** 2).sum())
                for s in range(0, len(x), SAE_CHUNK):  # bounds the (tokens x 16k) activations
                    xc = x[s:s + SAE_CHUNK]
                    f = saes[L].encode(xc)
                    st["tokens"] += len(xc)
                    st["l0_sum"] += float((f > 0).sum())
                    st["sq_err"] += float(((saes[L].decode(f) - xc) ** 2).sum())
                    pooled.index_add_(0, tok_article[s:s + SAE_CHUNK], f)
                pooled = (pooled / denom).to_sparse_csr()
                csr = sp.csr_matrix((pooled.values().cpu().numpy(), pooled.col_indices().cpu().numpy(),
                                     pooled.crow_indices().cpu().numpy()), shape=tuple(pooled.shape))
                sae_rows[L].append((state["idx"], csr))
            if L == n_layers - 1:
                raise _Stop  # skip the final norm and the 262k-vocab LM head
        return fn

    layers = decoder_layers(model, n_layers)
    handles = [layers[L].register_forward_hook(hook(L)) for L in range(n_layers)]
    t0, done = time.time(), 0
    for n_batch, b in enumerate(batches_by_tokens(lengths, max_tokens)):
        enc = tok([texts[i] for i in b], padding=True, return_tensors="pt").to(device)
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0  # drop BOS: its residual norm dwarfs every other token
        state.update(idx=b, mask=mask.float())
        try:
            model(**enc)
        except _Stop:
            pass
        done += len(b)
        if n_batch % 50 == 0:
            print(f"{name}: {done:,}/{N:,} articles, {time.time() - t0:.0f}s", flush=True)
    for h in handles:
        h.remove()
    for L, m in dense.items():
        np.save(out / f"dense_layer_{L}.npy", m)

    meta = {"model": model_id, "n_layers": n_layers, "d_model": d_model, "sae_repo": sae_repo,
            "sae_variant": SAE_VARIANT, "tokens_mean": float(lengths.mean() - 1), "tokens_max": int(lengths.max() - 1),
            "seconds": round(time.time() - t0), "sae": {}}
    for L in sae_layers:
        order = np.concatenate([i for i, _ in sae_rows[L]])
        mat = sp.vstack([m for _, m in sae_rows[L]]).tocsr()
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        mat = mat[inv]
        sp.save_npz(out / f"sae_layer_{L}.npz", mat)
        st = sae_stats[L]
        meta["sae"][str(L)] = {"width": saes[L].width, "token_l0": st["l0_sum"] / st["tokens"],
                               "fvu": st["sq_err"] / st["sq_tot"],
                               "article_nnz_mean": float(mat.getnnz(axis=1).mean())}
    return meta


def extract_sentence(name: str, texts: list[str], out: Path, device) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(SENTENCE[name], device=str(device))
    lengths = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
    limit = model.tokenizer.model_max_length
    if max(lengths) > limit:
        raise ValueError(f"longest article has {max(lengths)} tokens > {name} limit {limit}")
    model.max_seq_length = max(lengths)
    emb = model.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False)
    np.save(out / "dense.npy", emb.astype(np.float16))
    return {"model": SENTENCE[name], "dim": emb.shape[1], "tokens_max": max(lengths), "truncated": 0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=[*GEMMA, *SENTENCE], required=True)
    p.add_argument("--max-tokens", type=int, default=16_384, help="tokens per Gemma batch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="first N articles only (smoke tests)")
    p.add_argument("--out-root", type=Path, default=RAW_DIR)
    args = p.parse_args()
    out = args.out_root / args.model
    out.mkdir(parents=True, exist_ok=True)
    texts = article_text(load_processed_news())[:args.limit]
    device = torch.device(args.device)
    if args.model in GEMMA:
        meta = extract_gemma(args.model, texts, out, args.max_tokens, device)
    else:
        meta = extract_sentence(args.model, texts, out, device)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
