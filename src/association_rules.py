import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules


# ──────────────────────────────────────────
# 1. BASKET OLUŞTURMA
# ──────────────────────────────────────────

# Sepette her zaman bulunan, anlamsız kalemler
NOISE_ITEMS = {
    "POSTAGE", "DOTCOM POSTAGE", "CARRIAGE", "BANK CHARGES",
    "AMAZON FEE", "CRUK COMMISSION", "SAMPLES", "PACKING CHARGE",
    "MANUAL", "ADJUST", "CHECK", "TEST", "DISCOUNT"
}


def get_best_clustering_algorithm() -> tuple[str, str]:
    """En iyi kümeleme algoritmasını rapordan okur; dosya yoksa KMeans kullanır."""
    path = "outputs/reports/best_clustering_model.txt"
    label_map = {
        "kmeans": ("kmeans_label", "KMeans"),
        "k-means": ("kmeans_label", "KMeans"),
        "dbscan": ("dbscan_label", "DBSCAN"),
        "gmm": ("gmm_label", "GMM"),
    }

    try:
        with open(path, "r", encoding="utf-8") as file:
            raw_name = file.read().strip()
    except FileNotFoundError:
        raw_name = "KMeans"

    label_col, algo_name = label_map.get(raw_name.lower(), ("kmeans_label", "KMeans"))
    print(f"  Birliktelik analizi yalnızca en iyi algoritma için çalıştırılıyor: {algo_name}")
    return label_col, algo_name

def build_basket(df: pd.DataFrame) -> pd.DataFrame:
    # Anlamsız kalemleri temizle
    df = df[~df["Description"].str.upper().isin(NOISE_ITEMS)].copy()

    basket = (
        df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
        .astype(bool)
    )
    return basket


# ──────────────────────────────────────────
# 2. ZAMAN SERİSİ FİLTRESİ
# ──────────────────────────────────────────

def filter_recent_pairs(df: pd.DataFrame,
                         rules: pd.DataFrame,
                         months: int = 6) -> pd.DataFrame:
    """
    Son N ayda birlikte satılmamış ürün çiftlerini kurallardan çıkarır.
    """
    if rules.empty:
        return rules
    
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    cutoff = df["InvoiceDate"].max() - pd.DateOffset(months=months)
    recent = df[df["InvoiceDate"] >= cutoff]

    # Son N ayda hangi ürün çiftleri aynı faturada var?
    recent_basket = (
        recent.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum().unstack().fillna(0).astype(bool)
    )

    # Birlikte geçen çiftleri bul
    co_occurrence = set()
    cols = recent_basket.columns.tolist()
    arr  = recent_basket.values
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if (arr[:, i] & arr[:, j]).any():
                co_occurrence.add((cols[i], cols[j]))
                co_occurrence.add((cols[j], cols[i]))

    def is_recent(row):
        ants = list(row["antecedents"])
        cons = list(row["consequents"])
        for a in ants:
            for c in cons:
                if (a, c) in co_occurrence:
                    return True
        return False

    before = len(rules)
    rules  = rules[rules.apply(is_recent, axis=1)]
    print(f"      Zaman filtresi: {before} → {len(rules)} kural")
    return rules


# ──────────────────────────────────────────
# 3. HİBRİT PUANLAMA
# ──────────────────────────────────────────

def compute_hybrid_score(rules: pd.DataFrame,
                          w_confidence: float = 0.4,
                          w_lift: float = 0.6) -> pd.DataFrame:
    """
    Öneri Skoru = w1 * Confidence + w2 * Normalized_Lift
    Opsiyonel: Leverage ve Conviction da hesaplanır.
    """
    if rules.empty:
        return rules

    # Lift normalize et (0-1 arasına)
    lift_min = rules["lift"].min()
    lift_max = rules["lift"].max()
    if lift_max > lift_min:
        rules["lift_norm"] = (rules["lift"] - lift_min) / (lift_max - lift_min)
    else:
        rules["lift_norm"] = 1.0

    rules["hybrid_score"] = (
        w_confidence * rules["confidence"] +
        w_lift       * rules["lift_norm"]
    )

    return rules


# ──────────────────────────────────────────
# 4. KÜME İÇİ BASKINFLIK
# ──────────────────────────────────────────

def tag_rule_type(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Bir kural kaç farklı kümede yüksek lift ile çıkıyorsa:
    - 1 kümede → 'Personalized'
    - 2+ kümede → 'General'
    """
    if rules.empty or "segment" not in rules.columns:
        return rules

    cons_str = rules["consequents"].astype(str)
    ant_str  = rules["antecedents"].astype(str)
    rule_key = ant_str + "→" + cons_str

    counts = rule_key.groupby(rule_key).transform("count")
    rules["rule_type"] = counts.apply(
        lambda x: "General" if x > 1 else "Personalized"
    )
    return rules


# ──────────────────────────────────────────
# 5. KURAL ÇIKARMA (TÜM FİLTRELER)
# ──────────────────────────────────────────

def mine_rules(basket: pd.DataFrame,
               df_segment: pd.DataFrame,
               min_support: float = 0.02,
               min_lift: float = 1.2,       # Anlamlılık filtresi
               min_confidence: float = 0.5,
               apply_time_filter: bool = True) -> pd.DataFrame:

    MAX_COLS = 500
    if basket.shape[1] > MAX_COLS:
        top_cols = basket.sum().nlargest(MAX_COLS).index
        basket   = basket[top_cols]

    n_invoices = basket.shape[0]
    if n_invoices < 500:
        min_support = max(min_support, 0.05)
    elif n_invoices < 2000:
        min_support = max(min_support, 0.03)

    try:
        frequent = apriori(basket, min_support=min_support, use_colnames=True)
    except MemoryError:
        print("      ⚠️  Bellek yetersiz, min_support=0.10 ile tekrar deneniyor...")
        frequent = apriori(basket, min_support=0.10, use_colnames=True)

    if frequent.empty:
        return pd.DataFrame()

    rules = association_rules(
        frequent, metric="lift", min_threshold=min_lift,
        # Conviction ve Leverage dahil et
        num_itemsets=len(frequent)
    )

    # Temel filtreler
    rules = rules[rules["confidence"] >= min_confidence]

    # Conviction ve Leverage filtresi
    # leverage > 0  → birlikte satılma tesadüf değil
    # conviction > 1 → A, B olmadan daha az satılıyor
    if "leverage" in rules.columns:
        rules = rules[rules["leverage"] > 0]
    if "conviction" in rules.columns:
        rules = rules[rules["conviction"] > 1.0]

    if rules.empty:
        return pd.DataFrame()

    # Zaman serisi filtresi
    if apply_time_filter and not df_segment.empty:
        rules = filter_recent_pairs(df_segment, rules, months=6)

    if rules.empty:
        return pd.DataFrame()

    # Hibrit puanlama
    rules = compute_hybrid_score(rules)

    # hybrid_score'a göre sırala
    rules = rules.sort_values("hybrid_score", ascending=False)

    return rules


# ──────────────────────────────────────────
# 6. ALGORİTMA BAZLI ANALİZ
# ──────────────────────────────────────────

def run_rules_for_algorithm(df_train: pd.DataFrame,
                             label_col: str,
                             algo_name: str) -> pd.DataFrame:

    clustering = pd.read_csv("outputs/reports/clustering_results.csv")
    clustering = clustering[["CustomerID", label_col]].rename(
        columns={label_col: "segment"}
    )

    df = df_train.merge(clustering, on="CustomerID", how="inner")
    df = df[df["segment"] != -1]

    all_rules = []
    print(f"\n  [{algo_name}] Birliktelik Analizi:")

    for seg_id in sorted(df["segment"].unique()):
        seg_df = df[df["segment"] == seg_id]
        basket = build_basket(seg_df)

        print(f"    Segment {seg_id}: {len(seg_df)} işlem, "
              f"{basket.shape[1]} ürün")

        if basket.empty or basket.shape[1] < 2:
            print(f"      Segment {seg_id} atlandı: birliktelik analizi için yeterli ürün yok.")
            continue

        rules = mine_rules(basket, seg_df)

        if rules.empty:
            print(f"      ⚠️  Kural bulunamadı.")
            continue

        rules["segment"]   = seg_id
        rules["algorithm"] = algo_name

        # Küme baskınlık etiketi (segment bazlı)
        rules = tag_rule_type(rules)

        all_rules.append(rules)
        print(f"      ✅ {len(rules)} kural  "
              f"(General: {(rules.get('rule_type','') == 'General').sum()}, "
              f"Personalized: {(rules.get('rule_type','') == 'Personalized').sum()})")

    if all_rules:
        combined = pd.concat(all_rules, ignore_index=True)
        # Tüm segmentlerde kural sayısına göre General/Personalized yeniden etiketle
        combined = tag_rule_type(combined)
        path = f"outputs/reports/association_rules_{algo_name.lower()}.csv"
        combined.to_csv(path, index=False)
        print(f"    → {len(combined)} kural kaydedildi: {path}")
        return combined
    else:
        print(f"    ⚠️  {algo_name} için hiç kural üretilemedi.")
        return pd.DataFrame()


# ──────────────────────────────────────────
# 7. ANA FONKSİYON
# ──────────────────────────────────────────

def run_association_rules():
    df_train = pd.read_csv("data/processed/online_retail_train.csv")
    label_col, algo_name = get_best_clustering_algorithm()

    # DBSCAN/GMM gibi diğer algoritmalar için Apriori çalıştırılmaz;
    # gereksiz bellek tüketimini önlemek için yalnızca seçilen algoritma işlenir.
    rules_df = run_rules_for_algorithm(df_train, label_col, algo_name)

    if not rules_df.empty:
        rules_df.to_csv("outputs/reports/association_rules.csv", index=False)
        summary_rows = [{
            "algorithm":       algo_name,
            "n_rules":         len(rules_df),
            "avg_lift":        round(rules_df["lift"].mean(), 4),
            "avg_confidence":  round(rules_df["confidence"].mean(), 4),
            "avg_hybrid":      round(rules_df["hybrid_score"].mean(), 4),
            "max_lift":        round(rules_df["lift"].max(), 4),
            "n_general":       int((rules_df.get("rule_type", "") == "General").sum()),
            "n_personalized":  int((rules_df.get("rule_type", "") == "Personalized").sum()),
        }]
    else:
        summary_rows = [{
            "algorithm": algo_name, "n_rules": 0,
            "avg_lift": 0, "avg_confidence": 0,
            "avg_hybrid": 0, "max_lift": 0,
            "n_general": 0, "n_personalized": 0,
        }]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("outputs/reports/association_rules_comparison.csv",
                      index=False)

    print("\n  ╔═══════════╦════════╦══════════╦══════════╦══════════╗")
    print("  ║ Algoritma ║ Kural# ║ Ort.Lift ║ Ort.Conf ║ HybridSc ║")
    print("  ╠═══════════╬════════╬══════════╬══════════╬══════════╣")
    for _, r in summary_df.iterrows():
        print(f"  ║ {r['algorithm']:<9} ║ {int(r['n_rules']):<6} ║ "
              f"{r['avg_lift']:<8.4f} ║ {r['avg_confidence']:<8.4f} ║ "
              f"{r['avg_hybrid']:<8.4f} ║")
    print("  ╚═══════════╩════════╩══════════╩══════════╩══════════╝")
    print("\n  Birliktelik analizi tamamlandı.")
