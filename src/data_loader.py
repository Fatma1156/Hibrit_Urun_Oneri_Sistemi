import pandas as pd
from pathlib import Path

def load_raw_data():

    BASE_DIR = Path(__file__).resolve().parent.parent
    path = BASE_DIR / "data" / "raw" / "Online Retail.xlsx"

    df = pd.read_excel(path)

    print("Dataset yüklendi:", df.shape)

    return df