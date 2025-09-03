from clip_retrieval import clip_back
import fsspec
import os

def main():
    output_folder = os.environ.get("OUTPUT_FOLDER", "output_folder")
    indice_name = os.environ.get("INDICE_NAME", "example_index")
    backend_port = os.environ.get("BACKEND_PORT", "8000")

    index_folder = os.path.join(output_folder, "index")
    indice_path = os.path.join(output_folder, "indices_paths.json")

    with fsspec.open(indice_path, "w") as f:
        f.write(f'{{"{indice_name}": "{index_folder}"}}')

    clip_back(
        port=backend_port, 
        indices_paths=indice_path, 
        clip_model="ViT-B/32",
        enable_mclip_option=True, 
        provide_aesthetic_embeddings=False
    )
    

if __name__ == "__main__":
    main()