import json
import streamlit as st
import pandas as pd
import os
import torch
import tqdm

from models import SimpleModel
from utils import encode_input, get_output, decode_output, get_entropy

os.makedirs("models", exist_ok=True)
model_dirs = os.listdir("models")
model_dir = st.selectbox(
    "Choose a model to calculate entropy:",
    model_dirs,
)

os.makedirs("datasets", exist_ok=True)
datasets = os.listdir("datasets")
dataset_dir = st.selectbox(
    "Choose a dataset:",
    datasets,
)

if model_dir is None or dataset_dir is None:
    st.error("Initialize a model and dataset first")
    st.stop()

model = torch.load(os.path.join("models", model_dir, "model.pt"), weights_only=False)
unlabeled_df = pd.read_csv(os.path.join("datasets", dataset_dir, "unlabeled.tsv"), sep="\t", index_col=0)
labeled_df = pd.read_csv(os.path.join("datasets", dataset_dir, "labeled.tsv"), sep="\t", index_col=0)

base = os.path.join("models", model_dir)
input_dict_fn = os.path.join(base, "input_dict.json")
with open(input_dict_fn, "r") as f:
    input_dict = json.load(f)
output_dict_fn = os.path.join(base, "output_dict.json")
with open(output_dict_fn, "r") as f:
    output_alphabet = json.load(f)
output_reversal = dict((i, k) for (k, i) in output_alphabet.items())

def get_preds(model, df):
    nrows = df.shape[0]
    pbar = tqdm.tqdm(total=nrows)

    def forward_pass(row):
        word = row.word
        try:
            encoded = encode_input(word, input_dict)
            model_out = get_output(model, encoded, output_alphabet)
            entropy = get_entropy(model_out)
        except RuntimeError:
            entropy = 0
            model_out = None
        pbar.update(1)
        return model_out, entropy

    result = df.apply(forward_pass, axis=1, result_type='expand')
    result.columns = ['pred', 'entropy']
    reversal = dict((v, k) for (k, v) in output_alphabet.items())
    result.pred = result.pred.apply(lambda x: ";".join(decode_output(x, reversal)))
    return result

if st.button("Evaluate labeled data"):
    with st.spinner("Processing labeled data..."):
        preds = get_preds(model, labeled_df)

    labeled_df = labeled_df.join(preds)
    labeled_df.entropy = labeled_df.entropy.apply(float)
    labeled_df['correct'] = labeled_df.morphology == labeled_df.pred 
    st.markdown("## Model predictions on labeled data") 
    st.write(f"Accuracy: {labeled_df.correct.sum() / labeled_df.shape[0]:.1%} ({labeled_df.correct.sum()} out of {labeled_df.shape[0]})")
    st.write(labeled_df.sort_values(by='entropy', ascending=True))

frac = st.number_input("Fraction of unlabeled data to evaluate model on", value=0.1)

if st.button("Evaluate unlabeled data"):
    with st.spinner("Processing unlabeled data..."):
        preds = get_preds(model, unlabeled_df.sample(frac=frac))

    unlabeled_df = unlabeled_df.join(preds)
    unlabeled_df.entropy = unlabeled_df.entropy.apply(float)
    st.markdown("## Model predictions on unlabeled data") 
    st.write(unlabeled_df.sort_values(by='entropy', ascending=True))