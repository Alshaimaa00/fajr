from datasets import load_dataset
import pandas as pd

print("Loading dataset...")


dataset = load_dataset("riotu-lab/Quran-Tafseers")

df = dataset["train"].to_pandas()

print("First rows:")
print(df.head())

print("\nColumns:")
print(df.columns)


df.to_csv("quran_tafseers.csv", index=False, encoding="utf-8-sig")

print("\nDone! Dataset saved.")