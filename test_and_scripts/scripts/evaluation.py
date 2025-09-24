import json
import os

import pandas as pd


def to_list(x):
    if pd.isna(x):
        return []
    return [f.strip() for f in str(x).replace(',', ';').split(';') if f.strip()]


def load_annotations(path_excel: str, sheet_name: str = "Arkusz1") -> pd.DataFrame:
    ann = pd.read_excel(path_excel, sheet_name=sheet_name)
    ann = ann.rename(columns={
        "Anotacja": "query_id",
        "Nazwa zdjęcia - pliku": "relevant_images"
    })
    ann["relevant_images"] = ann["relevant_images"].apply(to_list)
    return ann.groupby("query_id").agg({
        "relevant_images": lambda x: sum(x, [])
    }).reset_index()


def load_results(path_csv: str) -> pd.DataFrame:
    res = pd.read_csv(path_csv, encoding="utf-8")
    res = res.rename(columns={
        "request": "query_id",
        "response": "retrieved_images",
        "time_sec": "query_time_sec"
    })
    res["retrieved_images"] = res["retrieved_images"].apply(json.loads)
    res["retrieved_images"] = res["retrieved_images"].apply(
        lambda lst: [d["image_path"] for d in lst]
    )
    return res.groupby("query_id").agg({
        "retrieved_images": lambda x: sum(x, []),
        "query_time_sec": "mean"
    }).reset_index()


def recall_precision_at_k(relevant: list, retrieved: list, k: int) -> tuple[float, float]:
    k = min(k, len(retrieved))
    retrieved_k = retrieved[:k]
    hits_k = len([img for img in retrieved_k if any(a in img for a in relevant)])
    recall_k = hits_k / len(relevant) if relevant else 0.0
    precision_k = hits_k / k if k else 0.0
    return recall_k, precision_k


def recall_at_50(relevant: list, retrieved: list) -> float:
    k = min(50, len(retrieved))
    retrieved_50 = retrieved[:k]
    hits_50 = len([img for img in retrieved_50 if any(a in img for a in relevant)])
    return hits_50 / len(relevant) if relevant else 0.0


def average_precision(relevant: list, retrieved: list) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precisions = []
    for idx, img in enumerate(retrieved, start=1):
        if any(a in img for a in relevant):
            hits += 1
            precisions.append(hits / idx)
    return sum(precisions) / len(relevant)


def compute_query_metrics(qid: str, relevant: list, retrieved: list, query_time_sec: float) -> dict:
    if not relevant:
        return None
    R = len(relevant)
    recall_K, precision_K = recall_precision_at_k(relevant, retrieved, R)
    recall_50 = recall_at_50(relevant, retrieved)
    AP = average_precision(relevant, retrieved)
    return {
        "query_id": qid,
        "Recall@K": recall_K,
        "Precision@K": precision_K,
        "Recall@50": recall_50,
        "AP": AP,
        "query_time_sec": query_time_sec
    }


def evaluate(ann_path, res_path, out_path):
    annotations = load_annotations(ann_path)
    results = load_results(res_path)
    metrics = []
    for _, row in annotations.iterrows():
        qid = row["query_id"]
        relevant = row["relevant_images"]
        if qid not in set(results["query_id"]):
            continue
        retrieved = list(results.loc[results.query_id == qid, "retrieved_images"].values[0])
        query_time_sec = float(results.loc[results.query_id == qid, "query_time_sec"].values[0])
        m = compute_query_metrics(qid, relevant, retrieved, query_time_sec)
        if m:
            metrics.append(m)
    df_metrics = pd.DataFrame(metrics)
    summary = {
        "mean_Recall@K": df_metrics["Recall@K"].mean(),
        "mean_Precision@K": df_metrics["Precision@K"].mean(),
        "mean_Recall@50": df_metrics["Recall@50"].mean(),
        "mAP": df_metrics["AP"].mean(),
        "mean_query_time_sec": df_metrics["query_time_sec"].mean()
    }
    print(df_metrics)

    print("\n=== Mean metrics ===")
    print(summary)
    os.makedirs(out_path, exist_ok=True)
    df_metrics.to_csv(f"{out_path}/metrics_per_query.csv", index=False, encoding="utf-8")
    pd.DataFrame([summary]).to_csv(f"{out_path}/metrics_summary.csv", index=False, encoding="utf-8")


def main():
    try:
        ann_path = "resources/Anotacje.xlsx"
        res_path = "resources/results.csv"
        out_path = "resources/output"
        evaluate(ann_path=ann_path, res_path=res_path, out_path=out_path)
    except Exception as e:
        print("Error occured:", e)


if __name__ == "__main__":
    main()
