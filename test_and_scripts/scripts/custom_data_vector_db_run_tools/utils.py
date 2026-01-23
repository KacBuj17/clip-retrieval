from functools import lru_cache

import numpy as np
import torch


def normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


@lru_cache(maxsize=1)
def load_clip_model(clip_model: str, mode: str = "text"):
    from all_clip import load_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if mode == "text":
        model, _, tokenizer = load_clip(clip_model, device=device)
        model.eval()
        return model, tokenizer, device
    elif mode == "image":
        model, preprocess, _ = load_clip(clip_model, device=device)
        model.eval()
        return model, preprocess, device
    else:
        raise ValueError(f"Unsupported mode: {mode}")


@lru_cache(maxsize=1)
def load_mclip_model(clip_model: str):
    from multilingual_clip import pt_multilingual_clip
    import transformers

    if clip_model == "ViT-L/14":
        name = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
    elif clip_model == "ViT-B/32":
        name = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
    else:
        raise ValueError(f"Unsupported CLIP model: {clip_model}")

    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(name)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)

    def encode(text: str) -> np.ndarray:
        with torch.no_grad():
            emb = model.forward([text], tokenizer)[0]
        return emb.cpu().numpy()

    return encode
