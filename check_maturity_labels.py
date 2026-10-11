import pandas as pd

CSV_PATH = r"E:\FreshLife_AI\auto_maturity_labels.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("MATURITY LABEL DISTRIBUTION")
print("=" * 60)

print("\nOverall:")
print(df["maturity_label"].value_counts())

print("\n\nFruit-wise distribution:")
print(
    pd.crosstab(
        df["original_class"],
        df["maturity_label"]
    )
)

print("\n\nAverage confidence:")
print(
    df.groupby("maturity_label")["confidence"]
    .mean()
    .round(2)
)

print("\nDone!")