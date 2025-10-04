import json
import os

import numpy as np
import pandas as pd
import seaborn as sns


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
        lambda x: list(dict.fromkeys(remove_ext(i) for i in to_list(x)))
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
        "retrieved_images": lambda x: list(dict.fromkeys(sum(x, []))),
        "query_time_sec": "mean"
    }).reset_index()
    df["retrieved_count"] = df["retrieved_images"].apply(len)
    return df


def recall_precision_at_k(rel, ret, k):
    k = min(k, len(ret))
    hits = set()
    for img in ret[:k]:
        new_hit = next((r for r in rel if r not in hits and r == img), None)
        if new_hit:
            hits.add(new_hit)
    tp = len(hits)
    recall = tp / len(rel) if rel else 0
    precision = tp / k if k else 0
    return recall, precision


def recall_at_50(rel, ret):
    hits = set()
    for img in ret[:50]:
        new_hit = next((r for r in rel if r not in hits and r == img), None)
        if new_hit:
            hits.add(new_hit)
    return len(hits) / len(rel) if rel else 0


def average_precision(rel, ret):
    hits = set()
    precisions = []
    for i, img in enumerate(ret, 1):
        new_hit = next((r for r in rel if r not in hits and r == img), None)
        if new_hit:
            hits.add(new_hit)
            precisions.append(len(hits) / i)
    return sum(precisions) / len(rel) if rel else 0


def compute_metrics(qid, rel, ret, time):
    if not rel:
        return None
    R = len(rel)
    rec, prec = recall_precision_at_k(rel, ret, R)
    return {
        "query_id": qid,
        "Recall@K": round(rec, 2),
        "Precision@K": round(prec, 2),
        "Recall@50": round(recall_at_50(rel, ret), 2),
        "AP": round(average_precision(rel, ret), 2),
        "retrieved_count": len(ret),
        "relevant_count": len(rel),
        "query_time_sec": round(time, 3)
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


def dynamic_k_threshold(relevant_count):
    if relevant_count <= 20:
        return 20
    elif relevant_count <= 50:
        return 50
    elif relevant_count <= 100:
        return 100
    else:
        return 150


def recall_precision_curve(rel, ret, k_max):
    recalls = []
    precisions = []
    for k in range(1, k_max + 1):
        rec, prec = recall_precision_at_k(rel, ret, k)
        recalls.append(round(rec, 2))
        precisions.append(round(prec, 2))
    return recalls, precisions


def plot_recall_precision(recalls, precisions, query_id, out_path, rel_num, model):
    sns.set_theme(style="darkgrid")
    k = list(range(1, len(recalls) + 1))
    df_plot = pd.DataFrame({
        "k": k,
        "Recall": recalls,
        "Precision": precisions
    })
    df_melt = df_plot.melt(id_vars="k", value_vars=["Recall", "Precision"],
                           var_name="Metric", value_name="Value")
    ax = sns.lineplot(data=df_melt, x="k", y="Value", hue="Metric", style="Metric",
                      markers=False, dashes=False, palette="tab10")

    title = f'Recall@K & Precision@K for query: "{query_id}"\n Model: {model}, Relevant images: {rel_num}'
    ax.set_title(title, wrap=True)

    fig = ax.get_figure()
    fig.subplots_adjust(top=0.85)
    fig.tight_layout()

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("K")
    ax.set_ylabel("Value")

    xticks = np.linspace(1, len(k), num=min(10, len(k)), dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)

    os.makedirs(f"{out_path}/k_results_plots/", exist_ok=True)
    fig = ax.get_figure()
    fig.savefig(f"{out_path}/k_results_plots/{query_id}_recall_precision.pdf", format='pdf')
    fig.savefig(f"{out_path}/k_results_plots/{query_id}_recall_precision.png", format='png')
    fig.clf()


def compute_plot_and_save_per_query(ann_path, res_path, out_path):
    ann = load_annotations(ann_path)
    res = load_results(res_path)
    all_metrics = []

    if "clip" in res_path.lower():
        model = "CLIP"
    elif "vlm_synonyms" in res_path.lower():
        model = "Paligemma with synonyms"
    elif "vlm" in res_path.lower():
        model = "Paligemma"

    for qid, rel_row in zip(ann["query_id"], ann["relevant_images"]):
        if qid not in set(res["query_id"]):
            continue
        ret_row = res.loc[res.query_id == qid, "retrieved_images"].values[0]
        k_max = dynamic_k_threshold(len(rel_row))
        recalls, precisions = recall_precision_curve(rel_row, ret_row, k_max)

        plot_recall_precision(recalls, precisions, qid, out_path, len(rel_row), model)

        df_query = pd.DataFrame({
            "query_id": [qid] * len(recalls),
            "k": list(range(1, len(recalls) + 1)),
            "Recall@k": recalls,
            "Precision@k": precisions
        })
        all_metrics.append(df_query)

    final_df = pd.concat(all_metrics, ignore_index=True)
    final_df.to_csv(f"{out_path}/k_results.csv", index=False, encoding="utf-8")


def main():
    ann_path = os.environ.get("ANN_PATH", "resources/Anotacje.xlsx")
    res_paths = ["resources/results_clip.csv", "resources/results_vlm.csv", "resources/results_vlm_synonyms.csv"]
    out_paths = ["resources/output_clip", "resources/output_vlm", "resources/output_vlm_synonyms"]

    for res_path, out_path in zip(res_paths, out_paths):
        metrics, mean_metrics = evaluate(ann_path, res_path)
        save_metrics(metrics, mean_metrics, out_path)

        compute_plot_and_save_per_query(ann_path, res_path, out_path)


if __name__ == "__main__":
    main()
