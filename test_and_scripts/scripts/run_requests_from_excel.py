import time
import json
import pandas as pd
from clip_retrieval.clip_client import ClipClient
import os

def main():
    backend_host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    backend_port = os.environ.get("BACKEND_PORT", "8000")
    url = f"http://{backend_host}:{backend_port}/knn-service"

    indice_name = os.environ.get("INDICE_NAME", "example_index")
    input_excel = os.environ.get("INPUT_EXCEL", "requests.xlsx")
    output_csv = os.environ.get("OUTPUT_CSV", "results.csv")
    num_images = int(os.environ.get("NUM_IMAGES", 5))

    df = pd.read_excel(input_excel)
    requests = df["Anotacja"].tolist()

    client = ClipClient(url=url, indice_name=indice_name, num_images=num_images)

    results_data = []
    for text in requests:
        try:
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
            "response": json.dumps(clean_results, ensure_ascii=False),
            "time_sec": time_sec
        })

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_csv, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()

