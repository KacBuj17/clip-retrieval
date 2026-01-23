import requests
import csv

URL = "http://127.0.0.1:5000/query"

def save_results_to_csv(results, filename="results.csv"):
    if not results:
        print("No results to save.")
        return

    keys = ["id", "score", "payload"]
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "id": r["id"],
                "score": r.get("score"),
                "payload": r.get("payload")
            })
    print(f"Results saved to {filename}")

def query_text(text, limit=5, vector_name="image", output_csv="text_results.csv"):
    payload = {
        "type": "text",
        "data": text,
        "vector_name": vector_name,
        "limit": limit
    }
    response = requests.post(URL, json=payload)
    if response.status_code == 200:
        results = response.json()["results"]
        save_results_to_csv(results, filename=output_csv)
    else:
        print(f"Error {response.status_code}: {response.text}")

def query_image(image_path, limit=5, vector_name="image", output_csv="image_results.csv"):
    payload = {
        "type": "image",
        "data": image_path,
        "vector_name": vector_name,
        "limit": limit
    }
    response = requests.post(URL, json=payload)
    if response.status_code == 200:
        results = response.json()["results"]
        save_results_to_csv(results, filename=output_csv)
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    query_text(text="cat", limit=3, vector_name="image", output_csv="cat_text_results.csv")
    query_image(image_path="example.jpg", limit=3, vector_name="image", output_csv="example_image_results.csv")
