from clip_retrieval import clip_back
import os

def main():
    resources_folder = os.environ.get("RESOURCES_FOLDER", "images_folder")
    output_folder = os.path.join(resources_folder, os.environ.get("OUTPUT_FOLDER", "output_folder"))
    backend_port = os.environ.get("BACKEND_PORT", "8000")

    indice_path = os.path.join(output_folder, "indices_paths.json")

    clip_back(
        port=backend_port, 
        indices_paths=indice_path, 
        clip_model="ViT-B/32",
        enable_mclip_option=True, 
        provide_aesthetic_embeddings=False
    )
    

if __name__ == "__main__":
    main()