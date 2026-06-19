import json
import streamlit as st
import pandas as pd
import os
import torch
import tqdm

from models import SimpleModel

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

base = os.path.join("models", model_dir)
input_dict_fn = os.path.join(base, "input_dict.json")
with open(input_dict_fn, "r") as f:
    input_dict = json.load(f)
output_dict_fn = os.path.join(base, "output_dict.json")
with open(output_dict_fn, "r") as f:
    output_alphabet = json.load(f)
output_reversal = dict((i, k) for (k, i) in output_alphabet.items())

def encode_input(word):
    idxs = []
    for c in word:
        try:
            idxs.append(input_dict[c])
        except KeyError:
            idxs.append(input_dict["UNK"])
    result = []
    for idx in idxs:
        t = torch.zeros(len(input_dict))
        t[idx] = 1
        result.append(t)
    result = torch.stack(result)
    if result.dim() == 1:
        result = result.unsqueeze(0)
    return result

def get_output(model, seq):
    result = [torch.Tensor([-99999] * len(output_alphabet))]
    result[0][output_alphabet["GLOSS_START"]] = 0
    MAX_LEN = 5
    with torch.no_grad():
        for _ in range(MAX_LEN):
            mask = torch.nn.Transformer.generate_square_subsequent_mask(len(result))
            next = model(seq, torch.stack(result), tgt_mask=mask, tgt_is_causal=True)
            argmax = torch.argmax(next[-1])
            result.append(next[-1])
            if argmax == output_alphabet["GLOSS_END"]:
                break
    
    return torch.stack(result)
    st.write([output_reversal[int(torch.argmax(y))] for y in result])

def get_entropy(log_probs):
    probs = torch.exp(log_probs)
    return torch.sum(probs * log_probs * -1)

nrows = unlabeled_df.shape[0]
pbar = tqdm.tqdm(total=nrows)

def forward_pass(word):
    try:
        encoded = encode_input(word)
        model_out = get_output(model, encoded)
        entropy = get_entropy(model_out)
    except RuntimeError:
        entropy = 0
    pbar.update(1)
    return entropy

def calc_entropy(model, df):
    result = df.word.apply(forward_pass)
    result.name = 'entropy'
    return result

if not st.button("Calculate entropy"):
    st.stop()

with st.spinner("Calculating entropy..."):
    entropy = calc_entropy(model, unlabeled_df)

st.write(unlabeled_df.join(entropy))