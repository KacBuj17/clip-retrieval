import os
import json
import pandas as pd


def to_list(x):
    if pd.isna(x):
        return []
    return [f.strip() for f in str(x).replace(',', ';').split(';') if f.strip()]


def remove_ext(path):
    return os.path.splitext(path)[0]


def load_annotations(path, sheet="Arkusz1"):
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.rename(columns={
        "Anotacja": "query_id",
        "Nazwa zdjęcia - pliku": "relevant_images"
    })
    df["relevant_images"] = df["relevant_images"].apply(
        lambda x: [remove_ext(i) for i in to_list(x)]
    )
    df = df.groupby("query_id", as_index=False).agg({
        "relevant_images": "sum"
    })
    df["relevant_count"] = df["relevant_images"].apply(len)
    return df

def load_results(path):
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={"request": "query_id",
                            "response": "retrieved_images",
                            "time_sec": "query_time_sec"})
    df["retrieved_images"] = df["retrieved_images"].apply(
        lambda x: [os.path.splitext(os.path.basename(d["image_path"]))[0]
                   for d in json.loads(x)]
    )
    df = df.groupby("query_id").agg({
        "retrieved_images": lambda x: list(set(sum(x, []))),
        "query_time_sec": "mean"
    }).reset_index()
    df["retrieved_count"] = df["retrieved_images"].apply(len)
    return df


def recall_precision_at_k(rel, ret, k):
    k = min(k, len(ret))
    hits = {r for r in rel if any(r in img for img in ret[:k])}
    return round(len(hits) / len(rel) if rel else 0, 2), round(len(hits) / k if k else 0, 2)


def recall_at_50(rel, ret):
    hits = {r for r in rel if any(r in img for img in ret[:50])}
    return round(len(hits) / len(rel) if rel else 0, 2)


def average_precision(rel, ret):
    hits, precisions = set(), []
    for i, img in enumerate(ret, 1):
        new = [r for r in rel if r not in hits and r in img]
        if new:
            hits.update(new)
            precisions.append(len(hits) / i)
    return round(sum(precisions) / len(rel), 2) if rel else 0


def compute_metrics(qid, rel, ret, time):
    if not rel:
        return None
    R = len(rel)
    rec, prec = recall_precision_at_k(rel, ret, R)
    return {
        "query_id": qid,
        "Recall@K": rec,
        "Precision@K": prec,
        "Recall@50": recall_at_50(rel, ret),
        "AP": average_precision(rel, ret),
        "retrieved_count": len(ret),
        "relevant_count": len(rel),
        "query_time_sec": round(time, 2)
    }


def compute_metrics_for_queries(ann_path, res_path):
    ann, res = load_annotations(ann_path), load_results(res_path)
    metrics = [
        compute_metrics(
            qid,
            ann_row,
            res.loc[res.query_id == qid, "retrieved_images"].values[0],
            res.loc[res.query_id == qid, "query_time_sec"].values[0]
        )
        for qid, ann_row in zip(ann["query_id"], ann["relevant_images"])
        if qid in set(res["query_id"])
    ]
    df = pd.DataFrame([m for m in metrics if m])
    return df


def compute_mean_metrics(df):
    summary = {f"mean_{col}": round(df[col].mean(), 2) for col in
               ["Recall@K", "Precision@K", "Recall@50", "AP", "query_time_sec"]}
    return summary


def evaluate(ann_path, res_path):
    metrics = compute_metrics_for_queries(ann_path, res_path)
    mean_metrics = compute_mean_metrics(metrics)
    return metrics, mean_metrics


def save_metrics(metrics, mean_metrics, out_path):
    os.makedirs(out_path, exist_ok=True)
    metrics.to_csv(f"{out_path}/metrics_per_query.csv", index=False, encoding="utf-8")
    pd.DataFrame([mean_metrics]).to_csv(f"{out_path}/metrics_summary.csv", index=False, encoding="utf-8")
    print(metrics, "\n=== Mean metrics ===\n", mean_metrics)


def read_and_fix_csv(path):
    df = pd.read_csv(path)
    df['request'] = df['request'].str.replace('\n', ' ').str.strip()


def main():
    ann_path = os.environ.get("ANN_PATH", "resources/Anotacje.xlsx")
    res_path = os.environ.get("RES_PATH", "resources/results.csv")
    out_path = os.environ.get("OUT_PATH", "resources/output_clip")

    read_and_fix_csv(res_path)
    metrics, mean_metrics = evaluate(ann_path, res_path)
    save_metrics(metrics, mean_metrics, out_path)


if __name__ == "__main__":
    main()
