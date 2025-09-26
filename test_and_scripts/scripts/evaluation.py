import json
import os

import pandas as pd


def to_list(x):
    if pd.isna(x):
        return []
    return [f.strip() for f in str(x).replace(',', ';').split(';') if f.strip()]


def remove_last_extension(path: str) -> str:
    return os.path.splitext(path)[0]


def load_annotations(path_excel: str, sheet_name: str = "Arkusz1") -> pd.DataFrame:
    ann = pd.read_excel(path_excel, sheet_name=sheet_name)
    ann = ann.rename(columns={
        "Anotacja": "query_id",
        "Nazwa zdjęcia - pliku": "relevant_images"
    })
    ann["relevant_images"] = ann["relevant_images"].apply(to_list)

    ann["relevant_images"] = ann["relevant_images"].apply(
        lambda lst: [remove_last_extension(x) for x in lst]
    )

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
        lambda lst: [remove_last_extension(d["image_path"]) for d in lst]
    )

    return res.groupby("query_id").agg({
        "retrieved_images": lambda x: sum(x, []),
        "query_time_sec": "mean"
    }).reset_index()


def recall_precision_at_k(relevant: list, retrieved: list, k: int) -> tuple[float, float]:
    k = min(k, len(retrieved))
    retrieved_k = retrieved[:k]
    hits_set = set()
    for r in relevant:
        if any(r in img for img in retrieved_k):
            hits_set.add(r)
    hits_k = len(hits_set)
    recall_k = hits_k / len(relevant) if relevant else 0.0
    precision_k = hits_k / k if k else 0.0
    return round(recall_k, 2), round(precision_k, 2)


def recall_at_50(relevant: list, retrieved: list) -> float:
    retrieved_50 = retrieved[:50]
    hits_set = set()
    for r in relevant:
        if any(r in img for img in retrieved_50):
            hits_set.add(r)
    return round(len(hits_set) / len(relevant), 2) if relevant else 0.0


def average_precision(relevant: list, retrieved: list) -> float:
    hits = set()
    precisions = []
    for idx, img in enumerate(retrieved, start=1):
        newly_hit = [r for r in relevant if r not in hits and r in img]
        if newly_hit:
            hits.update(newly_hit)
            precisions.append(len(hits) / idx)
    return round(sum(precisions) / len(relevant), 2)


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
        "query_time_sec": round(query_time_sec, 2)
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
        "mean_Recall@K": round(df_metrics["Recall@K"].mean(), 2),
        "mean_Precision@K": round(df_metrics["Precision@K"].mean(), 2),
        "mean_Recall@50": round(df_metrics["Recall@50"].mean(), 2),
        "mAP": round(df_metrics["AP"].mean(), 2),
        "mean_query_time_sec": round(df_metrics["query_time_sec"].mean(), 2)
    }
    print(df_metrics)

    print("\n=== Mean metrics ===")
    print(summary)
    os.makedirs(out_path, exist_ok=True)
    df_metrics.to_csv(f"{out_path}/metrics_per_query.csv", index=False, encoding="utf-8")
    pd.DataFrame([summary]).to_csv(f"{out_path}/metrics_summary.csv", index=False, encoding="utf-8")


def fix_csv_simple(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df['request'] = df['request'].str.replace('\n', ' ').str.strip()
    df.to_csv(output_path, index=False)


def main():
    try:
        ann_path = "resources/Anotacje.xlsx"
        res_path = "resources/results.csv"
        out_path = "resources/output_clip"
        fix_csv_simple(res_path, res_path)
        evaluate(ann_path=ann_path, res_path=res_path, out_path=out_path)
    except Exception as e:
        print("Error occured:", e)


if __name__ == "__main__":
    main()
