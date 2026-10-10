import pandas as pd
from pathlib import Path


def load_and_clean_data():
    df = pd.read_csv("data/raw/dataset.csv")
    df_clean = df.drop(columns=["CUST_ID"]).dropna()
    output_path = Path("data/processed/df_clean.csv")
    output_path.parent.mkdir(parents=True , exist_ok=True)
    df_clean.to_csv(output_path , index=False)
    return df_clean
