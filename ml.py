from datasets import load_dataset

emotions = load_dataset("emotion")

#print(emotions)
#print(emotions.column_names)
#print(emotions["train"].features)

import pandas as pd

emotions.set_format(type="pandas")
df = emotions["train"][:]
print(df.head())

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str)
print(df.head())

# looking at the class distribution
import matplotlib.pyplot as plt
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

# how long are the tweets
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False,
showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()