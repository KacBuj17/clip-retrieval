from clip_retrieval import clip_back
import os
import fsspec

def main():
    output_folder = "output_folder"

    index_folder = os.path.join(output_folder, "index")
    indice_path = os.path.join(output_folder, "indices_paths.json")

    with fsspec.open(indice_path, "w") as f:
        f.write('{"example_index": "' + index_folder + '"}')

    clip_back(
        port=1234, 
        indices_paths=indice_path, 
        clip_model="ViT-B/32",
        enable_mclip_option=True, 
        provide_aesthetic_embeddings=False
    )
    

if __name__ == "__main__":
    main()