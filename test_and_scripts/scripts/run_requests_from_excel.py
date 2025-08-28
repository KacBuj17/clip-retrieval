import time
import json
import pandas as pd
from clip_retrieval.clip_client import ClipClient

def main():
    url = 'http://127.0.0.1:1234/knn-service'
    indice_name = 'example_index'
    input_excel = "requests.xlsx"
    output_csv = "results.csv"

    df = pd.read_excel(input_excel)
    requests = df["Anotacja"].tolist()

    client = ClipClient(url=url, indice_name=indice_name)

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

