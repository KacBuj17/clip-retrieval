from functools import lru_cache

import numpy as np
import torch


def normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


@lru_cache(maxsize=1)
def load_clip_model(clip_model: str):
    from all_clip import load_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, tokenizer = load_clip(clip_model, device=device)
    model.eval()
    return model, tokenizer, device


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


class TextEmbedder:
    def __init__(self, clip_model="ViT-B/32", use_mclip=False):
        self.use_mclip = use_mclip
        self.clip_model = clip_model

        if use_mclip:
            self.encoder = load_mclip_model(clip_model)
        else:
            self.model, self.tokenizer, self.device = load_clip_model(clip_model)

    def encode(self, text: str) -> np.ndarray:
        if not text:
            raise ValueError("Text input is empty")

        if self.use_mclip:
            emb = self.encoder(text)
            if emb.ndim == 1:
                emb = emb[None, :]
            return normalize(emb)

        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()


def main():
    text = "a red car parked on the street"

    print("=== CLIP ===")
    clip_embedder = TextEmbedder(use_mclip=False)
    clip_vec = clip_embedder.encode(text)
    print("Shape:", clip_vec.shape)
    print(clip_vec[:1, :5])

    print("\n=== M-CLIP ===")
    mclip_embedder = TextEmbedder(use_mclip=True)
    mclip_vec = mclip_embedder.encode(text)
    print("Shape:", mclip_vec.shape)
    print(mclip_vec[:1, :5])


if __name__ == "__main__":
    main()
