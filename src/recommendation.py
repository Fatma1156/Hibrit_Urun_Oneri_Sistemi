import pandas as pd
import numpy as np
import joblib


def get_customer_segment(customer_id: int) -> int:
    segments = pd.read_csv("data/processed/customer_segments.csv",
                           index_col="CustomerID")
    if customer_id in segments.index:
        return int(segments.loc[customer_id, "segment"])
    return int(segments["segment"].value_counts().idxmax())


def _load_selected_features() -> list[str]:
    """Eğitimde kullanılan seçilmiş feature sırasını dosyadan okur."""
    with open("outputs/reports/selected_features.txt", "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def predict_segment_for_features(num_transactions: int,
                                  total_items: int,
                                  total_spent: float) -> int:
    """Yeni müşteriyi eğitimdeki scaler + K-Means pipeline'ı ile segmentler."""
    selected_features = _load_selected_features()
    scaler = joblib.load("outputs/models/scaler.pkl")
    kmeans = joblib.load("outputs/models/kmeans.pkl")

    row = pd.DataFrame(
        data=np.zeros((1, len(selected_features))),
        columns=selected_features
    )
    if "Frequency" in row.columns:
        row["Frequency"] = num_transactions
    if "total_items" in row.columns:
        row["total_items"] = total_items
    if "Monetary" in row.columns:
        row["Monetary"] = total_spent

    print("  Yeni müşteri feature değerleri:")
    print(row.to_string(index=False))

    X = scaler.transform(row[selected_features])
    segment = int(kmeans.predict(X)[0])
    print(f"  Tahmin edilen segment: {segment}")
    return segment


def parse_consequents(val) -> list:
    if isinstance(val, (set, frozenset)):
        return list(val)
    val = str(val)
    val = val.replace("frozenset(", "").replace(")", "")
    val = val.replace("{", "").replace("}", "")
    val = val.replace("'", "").replace('"', "")
    return [v.strip() for v in val.split(",") if v.strip()]


def recommend_products(customer_id: int = None,
                       segment_id: int = None,
                       top_n: int = 5,
                       already_bought: list = None,
                       algo_name: str = None) -> pd.DataFrame:
    """
    Hibrit öneri:
    - Segmentin birliktelik kurallarını hybrid_score'a göre sıralar
    - already_bought listesindeki ürünleri dışlar
    - General + Personalized kuralları dengeler
    """
    if segment_id is not None:
        segment = segment_id
    elif customer_id is not None:
        segment = get_customer_segment(customer_id)
    else:
        raise ValueError("customer_id veya segment_id verilmeli.")

    # Algoritma bazlı kural dosyası varsa onu kullan, yoksa genel
    if algo_name:
        path = f"outputs/reports/association_rules_{algo_name.lower()}.csv"
    else:
        path = "outputs/reports/association_rules.csv"

    try:
        rules = pd.read_csv(path)
    except FileNotFoundError:
        rules = pd.read_csv("outputs/reports/association_rules.csv")

    seg_rules = rules[rules["segment"] == segment].copy()

    if seg_rules.empty:
        print(f"  ⚠️  Segment {segment} için birliktelik kuralı bulunamadı.")
        return pd.DataFrame()

    # Consequents'i parse et
    seg_rules["product"] = seg_rules["consequents"].apply(
        lambda x: parse_consequents(x)[0] if parse_consequents(x) else ""
    )
    seg_rules = seg_rules[seg_rules["product"] != ""]

    # Satın alınanları çıkar
    if already_bought:
        bought_upper = [b.strip().upper() for b in already_bought]
        seg_rules = seg_rules[
            ~seg_rules["product"].str.upper().isin(bought_upper)
        ]

    # Hybrid score'a göre sırala (yoksa lift'e göre)
    sort_col = "hybrid_score" if "hybrid_score" in seg_rules.columns else "lift"
    seg_rules = seg_rules.sort_values(sort_col, ascending=False)
    seg_rules = seg_rules.drop_duplicates(subset="product")

    # General ve Personalized dengeleme: ilk yarı General, ikinci yarı Personalized
    if "rule_type" in seg_rules.columns:
        general      = seg_rules[seg_rules["rule_type"] == "General"].head(top_n // 2 + 1)
        personalized = seg_rules[seg_rules["rule_type"] == "Personalized"].head(top_n)
        combined     = pd.concat([general, personalized]).drop_duplicates(
            subset="product").head(top_n)
        if len(combined) < top_n:
            combined = seg_rules.head(top_n)
        seg_rules = combined

    cols = ["product", "confidence", "lift", "segment"]
    if "hybrid_score" in seg_rules.columns:
        cols.append("hybrid_score")
    if "rule_type" in seg_rules.columns:
        cols.append("rule_type")

    recommendations = seg_rules.head(top_n)[cols].reset_index(drop=True)
    recommendations.index += 1
    return recommendations


def _print_recs(recs: pd.DataFrame):
    if recs.empty:
        print("    Öneri bulunamadı.")
    else:
        for i, row in recs.iterrows():
            score = f"  hybrid={row['hybrid_score']:.3f}" \
                if "hybrid_score" in row else ""
            rtype = f"  [{row['rule_type']}]" \
                if "rule_type" in row else ""
            print(f"    {i}. {str(row['product']):<45} "
                  f"lift={row['lift']:.2f}  conf={row['confidence']:.2f}"
                  f"{score}{rtype}")


def run_recommendation_demo():
    print("\n  ===== ÖRNEK ÖNERİLER =====")

    print("\n  [A] Mevcut Müşteriler")
    segments = pd.read_csv("data/processed/customer_segments.csv",
                            index_col="CustomerID")
    sample_ids = segments.index[:3].tolist()

    for cid in sample_ids:
        seg = get_customer_segment(cid)
        print(f"\n  Müşteri {cid}  →  Segment {seg}")
        recs = recommend_products(customer_id=cid, top_n=5)
        _print_recs(recs)

    print("\n  [B] Yeni Müşteri Tahmini")
    new_customers = [
        {"num_transactions": 2,  "total_items": 15,  "total_spent": 45.0,
         "label": "Az Harcayan (Yeni)"},
        {"num_transactions": 20, "total_items": 300, "total_spent": 1800.0,
         "label": "Sık Alışveriş Yapan"},
        {"num_transactions": 5,  "total_items": 80,  "total_spent": 500.0,
         "label": "Orta Segment"},
    ]

    for nc in new_customers:
        seg = predict_segment_for_features(
            num_transactions=nc["num_transactions"],
            total_items=nc["total_items"],
            total_spent=nc["total_spent"]
        )
        print(f"\n  {nc['label']}  →  Tahmin Edilen Segment: {seg}")
        recs = recommend_products(segment_id=seg, top_n=5)
        _print_recs(recs)

    print("\n  Öneri sistemi testi tamamlandı.")
