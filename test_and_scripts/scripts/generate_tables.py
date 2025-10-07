import os
import pandas as pd
import yaml

def process_results(file_paths, uc_file, output_file_tex, output_dir_csv, k):
    with open(uc_file, "r", encoding="utf-8") as f:
        uc_map = yaml.safe_load(f)

    selected_columns = uc_map.get("selected_columns", None)
    model_names = ["VLM", "VLMs", "CLIP"]

    dfs = []
    base_df = pd.read_csv(file_paths[2])
    base_df = base_df[["query_id", "relevant_count"]].copy()
    base_df["relevant_count"] = base_df["relevant_count"].astype(int)
    dfs.append(base_df)

    for file_path, name in zip(file_paths, model_names):
        df = pd.read_csv(file_path)
        cols = ["query_id"] + [col for col in df.columns if col != "query_id" and col != "relevant_count"]
        if selected_columns is not None:
            cols = ["query_id"] + [col for col in selected_columns if col in df.columns]
        df = df[cols]
        new_cols = ["query_id"] + [f"{name}_{col}" for col in cols[1:]]
        df.columns = new_cols
        dfs.append(df.drop(columns=["query_id"]))

    merged = pd.concat(dfs, axis=1)
    merged = merged.round(2)

    def df_to_latex_table(df_uc, uc_name):
        mean_row = df_uc.mean(numeric_only=True).round(2)
        mean_row["query_id"] = "Average"
        for col in df_uc.columns:
            if "count" in col or "retrieved" in col:
                mean_row[col] = int(round(mean_row[col]))
        df_uc_with_mean = pd.concat([df_uc, pd.DataFrame([mean_row])], ignore_index=True)
        lines = []
        headers = [col.replace("_", "-") for col in df_uc_with_mean.columns]
        lines.append(" & ".join(headers) + r" \\")
        for _, row in df_uc_with_mean.iterrows():
            vals = [
                f"{row[col]:.2f}" if col not in ["query_id"] and "count" not in col and "retrieved" not in col
                else str(int(row[col])) if "count" in col or "retrieved" in col
                else str(row[col])
                for col in df_uc_with_mean.columns
            ]
            lines.append(" & ".join(vals) + r" \\")
        return f"% --- {uc_name} ---\n" + "\n".join(lines) + "\n"

    os.makedirs(output_dir_csv, exist_ok=True)

    with open(output_file_tex, "w", encoding="utf-8") as out_tex:
        for uc_name, queries in uc_map.items():
            if uc_name == "selected_columns":
                continue
            df_uc = merged[merged["query_id"].isin(queries)].copy()
            df_uc["order"] = df_uc["query_id"].apply(lambda q: queries.index(q) if q in queries else 999)
            df_uc = df_uc.sort_values("order").drop(columns=["order"])

            mean_row = df_uc.mean(numeric_only=True).round(2)
            mean_row["query_id"] = "Average"
            for col in df_uc.columns:
                if "count" in col or "retrieved" in col:
                    mean_row[col] = int(round(mean_row[col]))
            df_uc_with_mean = pd.concat([df_uc, pd.DataFrame([mean_row])], ignore_index=True)
            output_csv_path = os.path.join(output_dir_csv, f"{uc_name}-k{k}.csv")
            df_uc_with_mean.to_csv(output_csv_path, index=False)

            out_tex.write(df_to_latex_table(df_uc, uc_name))
            out_tex.write("\n\n")

        times_rows = []
        for file_path, name in zip(file_paths, model_names):
            df = pd.read_csv(file_path)
            avg_query = round(df["query_time_sec"].mean(), 4)
            avg_retrieved = round((df["query_time_sec"] / df["relevant_count"]).mean(), 4)
            times_rows.append([name, avg_query, avg_retrieved])

        lines = ["Times & Avg per query & Avg per retrieved \\\\"]
        for row in times_rows:
            lines.append(" & ".join([str(x) for x in row]) + r" \\")
        out_tex.write("% --- Times ---\n")
        out_tex.write("\n".join(lines) + "\n\n")

        times_df = pd.DataFrame(times_rows, columns=["Model", "Avg_per_query", "Avg_per_retrieved"])
        times_csv_path = os.path.join(output_dir_csv, f"times-k{k}.csv")
        times_df.to_csv(times_csv_path, index=False)

def main():
    k_list = [1, 2]
    uc_file = "tables.yaml"
    output_dir_csv = "resources/output_tables/csv"
    os.makedirs(output_dir_csv, exist_ok=True)

    for k in k_list:
        results_dir = f"resources/results-k-{k}"
        file_names = [
            "output_vlm/metrics_per_query.csv",
            "output_vlm_synonyms/metrics_per_query.csv",
            "output_clip/metrics_per_query.csv"
        ]
        file_paths = [os.path.join(results_dir, f) for f in file_names]
        output_file_tex = f"resources/output_tables/tables_uc-{k}.tex"
        process_results(file_paths, uc_file, output_file_tex, output_dir_csv, k)

if __name__ == "__main__":
    main()
