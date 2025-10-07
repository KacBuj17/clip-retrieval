import time
import json
import pandas as pd
from clip_retrieval.clip_client import ClipClient
import os


def to_list(x):
    if isinstance(x, str):
        return [i.strip() for i in x.split(",") if i.strip()]
    elif isinstance(x, list):
        return x
    else:
        return []


def remove_ext(filename):
    return os.path.splitext(filename)[0] if isinstance(filename, str) else filename


def load_annotations(path, sheet="Arkusz1"):
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.rename(columns={
        "Anotacja": "query_id",
        "Nazwa zdjęcia - pliku": "relevant_images"
    })
    df["relevant_images"] = df["relevant_images"].apply(
        lambda x: list(dict.fromkeys(remove_ext(i) for i in to_list(x)))
    )
    df = df.groupby("query_id", as_index=False).agg({
        "relevant_images": "sum"
    })
    df["relevant_count"] = df["relevant_images"].apply(len)
    return df

def dynamic_num_images_threshold(relevant_count):
    if relevant_count <= 10:
        return 20
    elif relevant_count <= 25:
        return 50
    elif relevant_count <= 50:
        return 100
    else:
        return 2 * relevant_count

def main():
    backend_host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    backend_port = os.environ.get("BACKEND_PORT", "8000")
    url = f"http://{backend_host}:{backend_port}/knn-service"

    resources_folder = os.environ.get("RESOURCES_FOLDER", "resources")
    indice_name = os.environ.get("INDICE_NAME", "example_index")

    input_excel = os.path.join(resources_folder, os.environ.get("INPUT_EXCEL", "Anotacje.xlsx"))
    output_csv = os.path.join(resources_folder, os.environ.get("OUTPUT_CSV", "results_clip.csv"))

    df = load_annotations(input_excel)
    requests = df[["query_id", "relevant_count"]].to_dict(orient="records")


    results_data = []
    for req in requests:
        text = req["query_id"]
        num_images = dynamic_num_images_threshold(req["relevant_count"])

        try:
            client = ClipClient(
                url=url,
                indice_name=indice_name,
                deduplicate=False,
                use_safety_model=False,
                use_violence_detector=False,
                num_images=num_images,
            )
            start = time.time()
            results = client.query(text=text)
            stop = time.time()
            time_sec = stop - start
        except Exception as e:
            results = {"error": str(e)}
            time_sec = None

        if isinstance(results, list):
            clean_results = []
            for r in results:
                if isinstance(r, dict):
                    r_clean = {k: v for k, v in r.items() if k != "image"}
                    clean_results.append(r_clean)
                else:
                    clean_results.append(r)
        elif isinstance(results, dict):
            clean_results = {k: v for k, v in results.items() if k != "image"}
        else:
            clean_results = results

        results_data.append({
            "request": text,
            "retrieved_count": num_images,
            "response": json.dumps(clean_results, ensure_ascii=False),
            "time_sec": time_sec
        })

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_csv, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
