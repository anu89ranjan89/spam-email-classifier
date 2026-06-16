import pandas as pd

data = []

with open("SMSSpamCollection", "r", encoding="utf-8") as file:
    for line in file:
        label, text = line.strip().split("\t", 1)
        data.append([label, text])

df = pd.DataFrame(data, columns=["label", "text"])

df.to_csv("spam.csv", index=False)

print("spam.csv created successfully!")