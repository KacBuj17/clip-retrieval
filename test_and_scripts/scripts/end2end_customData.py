import os
from clip_retrieval import clip_inference
from clip_retrieval import clip_index
from clip_retrieval import clip_back
import fsspec


def main(run_back=False):
    images_folder = os.environ.get("IMAGES_FOLDER", "images_folder")
    output_folder = os.environ.get("OUTPUT_FOLDER", "output_folder")

    fs, output_folder_in_fs = fsspec.core.url_to_fs(output_folder)
    print(output_folder_in_fs)
    if not fs.exists(output_folder_in_fs):
        fs.mkdir(output_folder_in_fs)
    embeddings_folder = os.path.join(output_folder, "embeddings")
    index_folder = os.path.join(output_folder, "index")

    clip_inference(
        input_dataset=images_folder,
        output_folder=embeddings_folder,
        input_format="files",
        enable_metadata=False,
        enable_text=False,
        write_batch_size=100000,
        batch_size=512,
        cache_path=None,
        clip_model="ViT-B/32",
        mclip_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        use_mclip=True,
    )
    
    os.mkdir(index_folder)
    clip_index(embeddings_folder, index_folder=index_folder)

    indice_path = os.path.join(output_folder, "indices_paths.json")
    with fsspec.open(indice_path, "w") as f:
        f.write('{"example_index": "' + index_folder + '"}')
    if run_back:
        clip_back(
            port=1234, 
            indices_paths=indice_path, 
            clip_model="ViT-B/32",
            enable_mclip_option=True, 
            provide_aesthetic_embeddings=False
        )


if __name__ == "__main__":
    main()
