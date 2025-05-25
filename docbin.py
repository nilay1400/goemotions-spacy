import pandas as pd
import spacy
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split

# Load your single-label CSV
df = pd.read_csv("goemotions_singlelabel.csv")
df = df.dropna(subset=["text", "emotion"])  # clean missing data

# List of all possible emotion labels (you can also extract from the dataset)
labels = sorted(df["emotion"].unique())  # or hardcode all 27 GoEmotion labels

# Train/dev split
train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42)

nlp = spacy.blank("en")


def df_to_docbin(dataframe, output_path):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc_bin = DocBin()
    for _, row in dataframe.iterrows():
        doc = nlp.make_doc(row["text"])
        cats = {label: 0.0 for label in labels}
        cats[row["emotion"]] = 1.0
        doc.cats = cats
        doc_bin.add(doc)
    doc_bin.to_disk(output_path)


# Save to .spacy
df_to_docbin(train_df, "data_spacy/train.spacy")
df_to_docbin(dev_df, "data_spacy/dev.spacy")

print("Converted to DocBin format.")
