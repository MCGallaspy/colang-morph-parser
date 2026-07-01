import json
import streamlit as st
import pandas as pd
import os
import torch
import tqdm

from models import SimpleModel
from utils import encode_input, get_output, get_entropy, decode_output

os.makedirs("models", exist_ok=True)
model_dirs = os.listdir("models")
model_dir = st.selectbox(
    "Choose a model to calculate entropy:",
    model_dirs,
)

os.makedirs("datasets", exist_ok=True)
datasets = os.listdir("datasets")
dataset_dir = st.selectbox(
    "Choose a word list:",
    datasets,
)

if model_dir is None or dataset_dir is None:
    st.error("Initialize a model and word list first")
    st.stop()

model_dict = torch.load(os.path.join("models", model_dir, "model.pt"))
model = SimpleModel(model_dict['d_input'], model_dict['num_glosses'])
model.load_state_dict(model_dict['weights'])
unlabeled_df = pd.read_csv(os.path.join("datasets", dataset_dir, "unlabeled.tsv"), sep="\t", index_col=0)

base = os.path.join("models", model_dir)
input_dict_fn = os.path.join(base, "input_dict.json")
with open(input_dict_fn, "r") as f:
    input_dict = json.load(f)
output_dict_fn = os.path.join(base, "output_dict.json")
with open(output_dict_fn, "r") as f:
    output_alphabet = json.load(f)
output_reversal = dict((i, k) for (k, i) in output_alphabet.items())

frac = st.number_input("Fraction of unlabeled words to calculate entropy on", value=0.1, format="%0.10g")

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
    result.pred = result.pred.apply(lambda x: decode_output(x, reversal))
    return result


if st.button("Calculate entropy"):
    with st.spinner("Calculating entropy..."):
        entropy = get_preds(model, unlabeled_df.sample(frac=frac))
    st.session_state['entropy'] = entropy
    st.session_state['labeled_df'] = pd.DataFrame(columns=["word", "morphology"])
    st.session_state['current_gloss'] = []

if st.session_state.get('entropy') is None:
    st.warning("You must calculate the entropy first")
    st.stop()

metadata_dict = os.path.join("datasets", dataset_dir, "metadata.json")
with open(metadata_dict, "r") as f:
    dataset_meta = json.load(f)

source_df = pd.read_csv(dataset_meta['source_file'], sep=dataset_meta['sep'])

entropy_df = unlabeled_df.join(st.session_state.entropy)
try:
    already_labeled_df = pd.read_csv(os.path.join("datasets", dataset_dir, "labeled.tsv"), sep="\t", index_col=0)
    entropy_df = entropy_df.drop(set(already_labeled_df.index) | set(st.session_state.labeled_df.index))
except:
    entropy_df = entropy_df.drop(st.session_state.labeled_df.index)
entropy_df = entropy_df.sort_values(by='entropy', ascending=False)
highest_entropy_row = entropy_df.iloc[0]

st.write(f"Total entropy = {entropy_df.entropy.sum():.1f}")
st.write(f"Average entropy = {entropy_df.entropy.mean():.1f}")

st.write(f"Highest entropy word: {highest_entropy_row.word} (entropy={highest_entropy_row.entropy:.2f})")
st.write(f"Source file data on word:")
st.write(source_df.loc[highest_entropy_row.name])
st.write("Label under construction:")
st.write(st.session_state.current_gloss)
st.write(f"Model guess: {';'.join(highest_entropy_row.pred)}")

if st.button("Use model guess"):
    st.session_state.current_gloss = highest_entropy_row.pred
    st.rerun()

gloss_element = st.selectbox(
    "Add Label element",
    list(output_alphabet.keys()),
)

if st.button("Add selected element"):
    st.session_state.current_gloss.append(gloss_element)
    st.rerun()

string_input = st.text_input("Add string input to label (characters must be in output dictionary)")
if st.button("Add string input") and string_input:
    if all(s in list(output_alphabet.keys()) for s in string_input):
        st.session_state.current_gloss += [s for s in string_input]
        st.rerun()
    else:
        st.warning("Not all characters in model's output alphabet")

if st.button("Add UNK"):
    st.session_state.current_gloss.append("UNK")
    st.rerun()

if st.button("Reset label"):
    st.session_state.current_gloss = []
    st.rerun()

if st.button("Finish label"):
    if st.session_state.current_gloss:
        st.session_state.labeled_df.loc[highest_entropy_row.name] = [
            highest_entropy_row.word,
            st.session_state.current_gloss,
        ]
        st.session_state.current_gloss = []
        st.rerun()

st.write("Newly labeled words")
mask = st.session_state.labeled_df.morphology.apply(lambda x: len(x) == 0)
st.session_state.labeled_df = st.session_state.labeled_df.loc[~mask]
st.write(st.session_state.labeled_df)

def add_gloss_start_end(glosses):
    if glosses[0] != "GLOSS_START":
        glosses = ["GLOSS_START"] + glosses
    if glosses[-1] != "GLOSS_END":
        glosses += ["GLOSS_END"]
    return glosses

if st.button("Update word list"):
    base = os.path.join("datasets", dataset_dir)
    labeled_df = st.session_state.labeled_df
    labeled_df["morphology"] = labeled_df["morphology"].apply(add_gloss_start_end).apply(lambda xs: ";".join(xs))
    try:
        already_labeled_df = pd.read_csv(os.path.join("datasets", dataset_dir, "labeled.tsv"), sep="\t", index_col=0)
        pd.concat([already_labeled_df, labeled_df]).to_csv(os.path.join(base, "labeled.tsv"), sep='\t')
    except:
        labeled_df.to_csv(os.path.join(base, "labeled.tsv"), sep='\t')
    st.session_state['labeled_df'] = pd.DataFrame(columns=["word", "morphology"])
    st.session_state.current_gloss = []
    st.success("Done!")