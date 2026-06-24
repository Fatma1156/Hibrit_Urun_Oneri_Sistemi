# Hibrit Ürün Öneri Sistemi

Online Retail veri seti üzerinde kümeleme + birliktelik kuralları tabanlı hibrit öneri sistemi.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python main.py
```

## Proje Yapısı

```
urun_oneri_sistemi/
├── main.py                  # Ana pipeline
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── Online Retail.xlsx   # Ham veri buraya konulmalı
│   └── processed/               # Otomatik oluşturulur
├── src/
│   ├── data_cleaning.py         # Veri temizleme
│   ├── data_loader.py           # Veri yükleme
│   ├── feature_engineering.py   # Özellik mühendisliği
│   ├── anomaly_detection.py     # Isolation Forest
│   ├── clustering_models.py     # K-Means / DBSCAN / GMM
│   ├── association_rules.py     # Segment bazlı Apriori
│   └── recommendation.py       # Hibrit öneri modülü
└── outputs/
    ├── models/                  # Eğitilmiş modeller (.pkl)
    ├── reports/                 # CSV çıktılar
    └── figures/                 # Grafikler
```

## Pipeline Adımları

1. **Veri Temizleme** — Eksik, iptalli, negatif kayıtlar ayıklanır
2. **Özellik Mühendisliği** — Müşteri bazlı işlem sayısı, miktar, harcama
3. **Anomali Tespiti** — Isolation Forest ile %2 anormal müşteri dışlanır
4. **Kümeleme** — K-Means, DBSCAN, GMM karşılaştırılır; en iyi Silhouette ile seçilir
5. **Birliktelik Analizi** — Her segment için Apriori çalıştırılır
6. **Öneri** — Müşteri segmenti → segment kuralları → lift sıralı top-N ürün
