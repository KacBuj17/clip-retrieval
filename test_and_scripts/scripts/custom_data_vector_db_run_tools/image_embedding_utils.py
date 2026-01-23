from functools import lru_cache

import numpy as np
import torch
from PIL import Image


def normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


@lru_cache(maxsize=1)
def load_clip_model(clip_model: str):
    from all_clip import load_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, _ = load_clip(clip_model, device=device)
    model.eval()
    return model, preprocess, device


class ImageEmbedder:
    def __init__(self, clip_model="ViT-B/32"):
        self.model, self.preprocess, self.device = load_clip_model(clip_model)

    def encode(self, image: Image.Image) -> np.ndarray:
        if image is None:
            raise ValueError("Image input is None")
        with torch.no_grad():
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


def main():
    image_path = "example.jpg"
    image = Image.open(image_path).convert("RGB")
    embedder = ImageEmbedder()
    vec = embedder.encode(image)
    print("Image embedding shape:", vec.shape)
    print(vec[:1, :5])


if __name__ == "__main__":
    main()
