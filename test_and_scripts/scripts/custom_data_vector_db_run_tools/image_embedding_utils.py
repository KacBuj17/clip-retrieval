import numpy as np
import torch
from PIL import Image

from utils import load_clip_model


class ImageEmbedder:
    def __init__(self, clip_model="ViT-B/32"):
        self.model, self.preprocess, self.device = load_clip_model(clip_model, mode="image")

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
