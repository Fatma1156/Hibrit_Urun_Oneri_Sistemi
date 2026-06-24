# ===============================
# ONLINE RETAIL — HİBRİT ÜRÜN ÖNERİ SİSTEMİ
# ===============================
# Pipeline:
#   1. EDA (ham veri üzerinde, sadece analiz)
#   2. Veri Temizleme
#   3. Feature Engineering (RFM + davranışsal özellikler + feature seçimi)
#   4. Anomali Tespiti (Isolation Forest)
#   5. Kümeleme → K-Means / DBSCAN / GMM
#   6. Segment Bazlı Birliktelik Kuralları (Apriori)
#   7. Hibrit Ürün Öneri Demo
#   8. Değerlendirme → Precision@K, Recall@K
# ===============================

from src.eda                 import run_eda
from src.data_cleaning       import clean_data
from src.feature_engineering import create_customer_features
from src.anomaly_detection   import run_anomaly_detection
from src.clustering_models   import run_clustering
from src.association_rules   import run_association_rules
from src.recommendation      import run_recommendation_demo
from src.evaluation          import run_evaluation
import os

def ensure_output_dirs():
    os.makedirs("outputs/reports", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

def main():
    ensure_output_dirs()
    
    print("\n" + "="*55)
    print("   HİBRİT ÜRÜN ÖNERİ SİSTEMİ — BAŞLADI")
    print("="*55 + "\n")

    # 1️⃣ EDA — ham veri üzerinde sadece analiz
    print("1) EDA — ham veri üzerinde keşifsel veri analizi...")
    run_eda()

    # 2️⃣ Veri Temizleme — kalıcı temizlik sadece burada yapılır
    print("\n2) Veri temizleme...")
    clean_data(remove_iqr_outliers=True)

    # 3️⃣ Feature Engineering — RFM + davranışsal özellikler + korelasyon seçimi
    print("\n3) Özellik mühendisliği ve feature seçimi...")
    features = create_customer_features()

    # 4️⃣ Anomali Tespiti
    print("\n4) Anomali tespiti (Isolation Forest)...")
    features_clean = run_anomaly_detection(features)

    # 5️⃣ Kümeleme
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
    run_evaluation(k=5)

    print("\n" + "="*55)
    print("   PROJE TAMAMLANDI ✅")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
