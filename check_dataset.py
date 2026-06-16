import pandas as pd

df = pd.read_csv("spam.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nLabels:")
print(df["label"].value_counts())