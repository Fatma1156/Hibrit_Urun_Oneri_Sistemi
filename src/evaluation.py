import pandas as pd
import numpy as np
from src.recommendation import predict_segment_for_features


def get_final_algorithm() -> tuple[str, str]:
    """Final kümeleme algoritmasını okur; dosya yoksa KMeans kullanır."""
    label_map = {
        "kmeans": ("KMeans", "kmeans_label"),
        "k-means": ("KMeans", "kmeans_label"),
        "dbscan": ("DBSCAN", "dbscan_label"),
        "gmm": ("GMM", "gmm_label"),
    }
    try:
        with open("outputs/reports/best_clustering_model.txt", "r", encoding="utf-8") as file:
            raw_name = file.read().strip()
    except FileNotFoundError:
        raw_name = "KMeans"

    return label_map.get(raw_name.lower(), ("KMeans", "kmeans_label"))


def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    if not recommended or not relevant:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / k


def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    if not recommended or not relevant:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)


def parse_consequents(val) -> list:
    if isinstance(val, (set, frozenset)):
        return list(val)
    val = str(val)
    val = val.replace("frozenset(", "").replace(")", "")
    val = val.replace("{", "").replace("}", "")
    val = val.replace("'", "").replace('"', "")
    return [v.strip() for v in val.split(",") if v.strip()]


def get_segment_for_customer(cid, label_col, clustering_df,
                              test_features, segments):
    """Müşterinin belirli algoritmaya göre segment ID'sini döndürür."""
    row = clustering_df[clustering_df["CustomerID"] == cid]
    if not row.empty:
        seg = row.iloc[0][label_col]
        if seg != -1:
            return int(seg)

    # Clustering'de yoksa özelliklerden tahmin et
    if not test_features.empty and cid in test_features.index:
        r = test_features.loc[cid]
        return predict_segment_for_features(
            num_transactions=int(r.get("Frequency", 1)),
            total_items=int(r.get("total_items", 1)),
            total_spent=float(r.get("Monetary", 0.0))
        )

    # Fallback: en kalabalık segment
    valid = clustering_df[clustering_df[label_col] != -1]
    return int(valid[label_col].value_counts().idxmax())


def evaluate_algorithm(algo_name: str,
                        label_col: str,
                        train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        clustering_df: pd.DataFrame,
                        test_features: pd.DataFrame,
                        segments: pd.DataFrame,
                        k: int = 5,
                        sample_size: int = 200,
                        random_state: int = 42) -> dict:
    """
    Tek bir algoritma için Precision@K ve Recall@K hesaplar.
    """
    # O algoritmaya ait birliktelik kurallarını yükle
    rules_path = f"outputs/reports/association_rules_{algo_name.lower()}.csv"
    try:
        rules_df = pd.read_csv(rules_path)
    except FileNotFoundError:
        print(f"    ⚠️  {rules_path} bulunamadı, atlanıyor.")
        return {}

    rules_df["consequents_parsed"] = rules_df["consequents"].apply(
        lambda x: parse_consequents(x)[0] if parse_consequents(x) else ""
    )

    test_customers = test_df["CustomerID"].unique().tolist()
    if len(test_customers) > sample_size:
        rng = np.random.default_rng(random_state)
        test_customers = rng.choice(
            test_customers, size=sample_size, replace=False
        ).tolist()

    results = []

    for cid in test_customers:
        seg = get_segment_for_customer(
            cid, label_col, clustering_df, test_features, segments
        )

        bought_train = (
            train_df[train_df["CustomerID"] == cid]["Description"]
            .dropna().str.strip().str.upper().unique().tolist()
        )
        bought_test = (
            test_df[test_df["CustomerID"] == cid]["Description"]
            .dropna().str.strip().str.upper().unique().tolist()
        )

        if not bought_test:
            continue

        seg_rules = rules_df[rules_df["segment"] == seg].copy()
        seg_rules = seg_rules[
            ~seg_rules["consequents_parsed"].str.upper().isin(bought_train)
        ]
        seg_rules = seg_rules.sort_values("lift", ascending=False)
        seg_rules = seg_rules.drop_duplicates(subset="consequents_parsed")

        recommended = [
            r.strip().upper()
            for r in seg_rules["consequents_parsed"].head(k).tolist()
        ]

        results.append({
            "CustomerID":      cid,
            "segment":         seg,
            f"Precision@{k}":  precision_at_k(recommended, bought_test, k),
            f"Recall@{k}":     recall_at_k(recommended, bought_test, k),
            "n_recommended":   len(recommended),
            "n_relevant":      len(bought_test),
        })

    if not results:
        return {"algorithm": algo_name, f"Precision@{k}": 0,
                f"Recall@{k}": 0, "n_evaluated": 0}

    df_res = pd.DataFrame(results)
    df_res.to_csv(
        f"outputs/reports/evaluation_{algo_name.lower()}.csv", index=False
    )

    p_col, r_col = f"Precision@{k}", f"Recall@{k}"
    return {
        "algorithm":           algo_name,
        f"Precision@{k}":      round(df_res[p_col].mean(), 4),
        f"Recall@{k}":         round(df_res[r_col].mean(), 4),
        "n_evaluated":         len(df_res),
        "n_with_recs":         int((df_res["n_recommended"] > 0).sum()),
    }


def run_evaluation(k: int = 5, sample_size: int = 200):
    algo_name, label_col = get_final_algorithm()
    print(f"\n  Değerlendirme yalnızca final algoritma için yapılıyor: {algo_name}")

    train_df = pd.read_csv("data/processed/online_retail_train.csv")
    test_df  = pd.read_csv("data/processed/online_retail_test.csv")

    clustering_df = pd.read_csv("outputs/reports/clustering_results.csv")

    segments = pd.read_csv("data/processed/customer_segments.csv",
                            index_col="CustomerID")

    try:
        test_features = pd.read_csv(
            "data/processed/customer_features_test.csv",
            index_col="CustomerID"
        )
    except FileNotFoundError:
        test_features = pd.DataFrame()

    result = evaluate_algorithm(
        algo_name, label_col,
        train_df, test_df,
        clustering_df, test_features, segments,
        k=k, sample_size=sample_size
    )

    if not result:
        print("  ⚠️  Final algoritma değerlendirilemedi.")
        return {}

    summary_df = pd.DataFrame([result])
    summary_df.to_csv("outputs/reports/evaluation_comparison.csv", index=False)

    p_col, r_col = f"Precision@{k}", f"Recall@{k}"

    print(f"\n  {'='*55}")
    print(f"  FINAL ALGORİTMA DEĞERLENDİRMESİ (K={k})")
    print(f"  {'='*55}")
    print(f"  {'Algoritma':<10} {'Precision@'+str(k):<16} "
          f"{'Recall@'+str(k):<16} {'Değerlendirilen'}")
    print(f"  {'-'*55}")
    for _, row in summary_df.iterrows():
        print(f"  {row['algorithm']:<10} {row[p_col]:<16.4f} "
              f"{row[r_col]:<16.4f} {int(row['n_evaluated'])}")

    final = summary_df.iloc[0]
    print(f"\n  ✅ Final algoritma: {final['algorithm']} "
          f"(Precision@{k}={final[p_col]:.4f}, "
          f"Recall@{k}={final[r_col]:.4f})")
    print(f"  {'='*55}\n")

    print("  Değerlendirme tamamlandı.")
    return summary_df.to_dict("records")
