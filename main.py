# ===============================
# ONLINE RETAIL — HİBRİT ÜRÜN ÖNERİ SİSTEMİ
# ===============================
# Pipeline:
#   1. Veri Temizleme
#   2. EDA (Keşif + Description Temizliği + Aykırı Değer)
#   3. Gelişmiş Feature Engineering (RFM + ekstra özellikler)
#   4. Anomali Tespiti (Isolation Forest)
#   5. Kümeleme → K-Means / DBSCAN / GMM + 3 metrik karşılaştırması
#   6. Segment Bazlı Birliktelik Kuralları (Apriori)
#   7. Hibrit Ürün Öneri Demo (mevcut + yeni müşteri)
#   8. Değerlendirme → Precision@K, Recall@K
# ===============================

from src.data_cleaning       import clean_data
from src.eda                 import run_cleaning_with_eda, run_eda
from src.feature_engineering import create_customer_features
from src.anomaly_detection   import run_anomaly_detection
from src.clustering_models   import run_clustering
from src.association_rules   import run_association_rules
from src.recommendation      import run_recommendation_demo
from src.evaluation          import run_evaluation


def main():

    print("\n" + "="*55)
    print("   HİBRİT ÜRÜN ÖNERİ SİSTEMİ — BAŞLADI")
    print("="*55 + "\n")

    # 1️⃣ Veri Temizleme
    print("1) Veri temizleme...")
    clean_data()

    # 2️⃣ EDA — Description temizliği + Aykırı değer + Görseller
    print("\n2) EDA — Keşifsel veri analizi...")
    run_cleaning_with_eda()
    run_eda()

    # 3️⃣ Gelişmiş Feature Engineering (RFM + 8 özellik)
    print("\n3) Özellik mühendisliği (RFM + gelişmiş)...")
    features = create_customer_features()

    # 4️⃣ Anomali Tespiti
    print("\n4) Anomali tespiti (Isolation Forest)...")
    features_clean = run_anomaly_detection(features)

    # 5️⃣ Kümeleme (K-Means / DBSCAN / GMM → 3 metrik karşılaştırması)
    print("\n5) Kümeleme modelleri karşılaştırılıyor...")
    run_clustering(features_clean)

    # 6️⃣ Segment Bazlı Birliktelik Kuralları
    print("\n6) Segment bazlı birliktelik analizi...")
    run_association_rules()

    # 7️⃣ Ürün Öneri Demo
    print("\n7) Ürün öneri sistemi test ediliyor...")
    run_recommendation_demo()

    # 8️⃣ Değerlendirme
    print("\n8) Model değerlendirmesi (Precision@K, Recall@K)...")
    run_evaluation(k=5, sample_size=200)

    print("\n" + "="*55)
    print("   PROJE TAMAMLANDI ✅")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
