import numpy as np
import torch

from utils import load_clip_model, load_mclip_model, normalize


class TextEmbedder:
    def __init__(self, clip_model="ViT-B/32", use_mclip=False):
        self.use_mclip = use_mclip
        self.clip_model = clip_model

        if use_mclip:
            self.encoder = load_mclip_model(clip_model)
        else:
            self.model, self.tokenizer, self.device = load_clip_model(clip_model, mode="text")

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
