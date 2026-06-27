import json
import streamlit as st
import pandas as pd
import os
import torch
import torch.nn as nn
import tqdm

from matplotlib import pyplot as plt
from models import SimpleModel

os.makedirs("models", exist_ok=True)
model_dirs = os.listdir("models")
model_dir = st.selectbox(
    "Choose a model to train:",
    model_dirs,
)

os.makedirs("datasets", exist_ok=True)
datasets = os.listdir("datasets")
dataset_dir = st.selectbox(
    "Choose a word list to train on:",
    datasets,
)

if model_dir is None or dataset_dir is None:
    st.error("Initialize a model and dataset first")
    st.stop()

model_dict = torch.load(os.path.join("models", model_dir, "model.pt"))
model = SimpleModel(model_dict['d_input'], model_dict['num_glosses'])
model.load_state_dict(model_dict['weights'])
training_df = pd.read_csv(os.path.join("datasets", dataset_dir, "labeled.tsv"), sep="\t", index_col=0)

base = os.path.join('models', model_dir)
input_dict_fn = os.path.join(base, "input_dict.json")
with open(input_dict_fn, "r") as f:
    input_alphabet = json.load(f)
output_dict_fn = os.path.join(base, "output_dict.json")
with open(output_dict_fn, "r") as f:
    output_dict = json.load(f)

N = len(input_alphabet)
pbar = tqdm.tqdm(total=N)

def get_input_sequence(word):
    idxs = []
    for c in word:
        try:
            idxs.append(input_alphabet[c])
        except KeyError:
            idxs.append(input_alphabet['OOV'])
    result = []
    for idx in idxs:
        t = torch.zeros(N)
        t[idx] = 1
        result.append(t)
    result = torch.stack(result)
    return result


training_df['input_sequence'] = training_df.word.apply(get_input_sequence)

def get_output_sequence(gloss):
    gloss = [g.strip() for g in gloss.split(";")]
    seq_len = len(gloss)
    output_dim = len(output_dict)
    y = torch.Tensor([[torch.iinfo(torch.int16).min] * output_dim] * seq_len)
    for seq_idx, g in enumerate(gloss):
        try:
            gloss_idx = output_dict[g]
        except KeyError:
            gloss_idx = output_dict["UNK"]
        y[seq_idx][gloss_idx] = 0
    return y

training_df['output_sequence'] = training_df.morphology.apply(get_output_sequence)

losses = []
num_epochs = st.number_input("Number of epochs", value=10)

learning_rate = st.number_input("Learning rate", value=1e-4, format="%0.2e")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.NLLLoss(reduction='sum')

if st.button("Train"):
    with st.spinner("Training..."):
        model.train()
        for n in tqdm.tqdm(range(num_epochs)):
            epoch_loss = 0
            for row in training_df.sample(frac=1.0).itertuples():
                X = row.input_sequence
                y = row.output_sequence[:-1, :]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(y.shape[0])
                pred = model(X, torch.exp(y), tgt_mask=tgt_mask, tgt_is_causal=True)
                actual_labels = torch.argmax(row.output_sequence[1:], axis=1)
                loss = criterion(pred, actual_labels)
                epoch_loss += loss 
            epoch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(epoch_loss.detach().item() / training_df.shape[0])

        torch.save({
            "d_input": model.d_input,
            "num_glosses": model.num_glosses,
            "weights": model.state_dict(),
        }, os.path.join(base, "model.pt"))
    st.success("Model weights saved!")

    fig = plt.figure()
    MAX_PLOT_POINTS = 1000
    if len(losses) > MAX_PLOT_POINTS:
        step = len(losses) // MAX_PLOT_POINTS
        begin, end = losses[0], losses[-1]
        losses = [begin] + losses[::step] + [end]
    pd.Series(losses).plot(ax=plt.gca())
    st.pyplot(fig)