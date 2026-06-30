import streamlit as st
import pandas as pd
import os
import json
import torch

from itertools import product

from models import SimpleModel

st.subheader("Load list of words")
words_file = st.file_uploader("Upload a tsv with words and their morphology")
sep = st.text_input("Separator character", value=r"\t")
if words_file is not None:
    words_filename = words_file.name
    df = pd.read_csv(words_file, sep=sep)
    st.header("File preview")
    st.write(df.head())
else:
    st.error("Select a words file to continue")
    st.stop()

cols = list(df)
words_col = st.selectbox(
    "Which column has input words?",
    cols,
)

morph_col = st.selectbox(
    "Which column (if any) has the morphology?",
    ["None"] + cols,
)

if morph_col != "None":
    num_glossed = df[morph_col].isna().sum()
    print("num glossed", num_glossed, " / ", df.shape[0])
    if num_glossed == 0:
        st.warning("Morphology is given for all words already!"
                   " Add some unlabeled words to the word list.")

st.subheader("Instantiate word list")
dataset_name = st.text_input("Word list name", value="my cool word list")

fraction = st.number_input("Select a fraction of the unlabeled words to input", value=1.0)

if st.button("Instantiate!"):
    base = os.path.join("datasets", dataset_name)
    os.makedirs(base, exist_ok=True)
    
    if morph_col != "None":
        mask = ~df[morph_col].isna()
        
        tmp = df.loc[mask, [words_col, morph_col]]
        tmp.columns = ["word", "morphology"]
        tmp.to_csv(os.path.join(base, "labeled.tsv"), sep='\t')
        
        tmp = df.loc[~mask, words_col]
        tmp.name = "word"
        tmp.sample(frac=fraction).to_csv(os.path.join(base, "unlabeled.tsv"), sep='\t')
    else:
        tmp = df.loc[:, words_col]
        tmp.name = "word"
        print(tmp.head())
        tmp.sample(frac=fraction).to_csv(os.path.join(base, "unlabeled.tsv"), sep='\t')
    
    metadata_dict = os.path.join(base, "metadata.json")
    with open(metadata_dict, "w") as f:
        json.dump({
            "source_file": os.path.realpath(words_filename),
            "sep": sep,
        }, f)
    
    st.success("Initialized!")
