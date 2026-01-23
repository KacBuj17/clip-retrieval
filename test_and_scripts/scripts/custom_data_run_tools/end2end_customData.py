import json
import os
import sys
import time

import fsspec
import yaml

from clip_retrieval import clip_back
from clip_retrieval import clip_index
from clip_retrieval import clip_inference


def main(run_back=False):
    resources_folder = os.environ.get("RESOURCES_FOLDER", "resources")
    images_folder = os.path.join(resources_folder, os.environ.get("IMAGES_FOLDER", "images_folder"))
    output_folder = os.path.join(resources_folder, os.environ.get("OUTPUT_FOLDER", "output_folder"))

    with open("db_config.yaml", "r") as f:
        db_config = yaml.safe_load(f)

    fs, output_folder_in_fs = fsspec.core.url_to_fs(output_folder)
    print(output_folder_in_fs)
    if not fs.exists(output_folder_in_fs):
        fs.mkdir(output_folder_in_fs)
    embeddings_folder = os.path.join(output_folder, "embeddings")
    index_folder = os.path.join(output_folder, "index")

    inference_start_time = time.time()
    clip_inference(
        input_dataset=images_folder,
        output_folder=embeddings_folder,
        input_format="files",
        enable_metadata=True,
        enable_text=True,
        write_batch_size=100000,
        batch_size=512,
        cache_path=None,
        clip_model="ViT-B/32",
        mclip_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        use_mclip=True,
        writer_type=db_config["writer_type"],
        mongo_uri=db_config["mongo_uri"],
        mongo_db_name=db_config["mongo_db_name"],
        qdrant_url=db_config["qdrant_url"],
        qdrant_api=db_config["qdrant_api"],
        collection_name=db_config["collection_name"],
    )
    inference_stop_time = time.time()
    inference_duration = inference_stop_time - inference_start_time

    os.mkdir(index_folder)

    index_start_time = time.time()
    clip_index(embeddings_folder, index_folder=index_folder)
    index_stop_time = time.time()
    index_duration = index_stop_time - index_start_time

    stats = {
        "inference_duration": inference_duration,
        "index_duration": index_duration
    }

    output_file = os.path.join(resources_folder, "stats.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

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
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}", file=sys.stderr)
        sys.exit(1)
