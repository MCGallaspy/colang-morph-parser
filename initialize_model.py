import streamlit as st
import pandas as pd
import os
import json
import torch

from itertools import product

from models import SimpleModel

st.subheader("Input specification")
words_file = st.file_uploader("Upload a tsv with words to analyze")
sep = st.text_input("Separator character", value=r"\t")
if words_file is not None:
    df = pd.read_csv(words_file, sep=sep)
    st.header("File preview")
    st.write(df.head())
else:
    st.stop("Select a words file to continue")

cols = list(df)
words_col = st.selectbox(
    "Which column has input words?",
    cols,
)
specials = st.text_input("Comma-separated list of special input tokens (e.g. PAD or UNK)")
specials = [s.strip() for s in specials.split(",")]
specials = [s for s in specials if s]
input_alphabet = set()
if words_col:
    for word in df[words_col].values:
        input_alphabet |= set(word)
    input_alphabet |= set(specials)
    input_alphabet = dict((w, i) for (i, w) in enumerate(input_alphabet))
    d_input = len(input_alphabet)
    st.write(f"Input dimensionality: {d_input}")
    st.write("Input alphabet")
    st.write(input_alphabet)


st.subheader("Output specification")
if st.button("Reset output vocab") and 'output_vocab' in st.session_state:
    del st.session_state['output_vocab']

if 'output_vocab' not in st.session_state:
    st.session_state['output_vocab'] = pd.DataFrame(data=None, columns=["tokens",], dtype=str)

output_tokens = set(st.session_state.output_vocab.tokens)

st.markdown("**Inflectional element builder**")
st.text("Create output tokens that are the Cartesian product of the following sets of symbols. Enter as comma-separated lists.")
inflectional_sets = []
for i in range(4):
    elements = st.text_input(f"Set {i}", key=f"set_{i}")
    elements = [s.strip() for s in elements.split(",")]
    elements = [s for s in elements if s]
    if elements:
        inflectional_sets.append(elements)

if any(inflectional_sets):
    to_add = [".".join(e) for e in product(*inflectional_sets)]
    st.write("Inflectional elements")
    st.write(to_add)
    if st.button("Add these elements"):
        output_tokens |= set(to_add)

st.markdown("**Special output tokens**")
output_specials = st.text_input(
    "Comma-separated list of special output tokens",
    value="GLOSS_START, GLOSS_END",
)
output_specials = [s.strip() for s in output_specials.split(",")]
output_specials = [s for s in output_specials if s]
output_tokens |= set(output_specials)

st.markdown("**Output vocab**")
indexes = list(range(len(output_tokens)))
st.session_state.output_vocab = st.session_state.output_vocab.reindex(index=indexes)
st.session_state.output_vocab.loc[indexes, "tokens"] = pd.Series(list(output_tokens))
st.session_state.output_vocab = st.data_editor(st.session_state.output_vocab, num_rows="dynamic")

st.subheader("Instantiate model")
st.text("Once input and output specification are complete, instantiate and save the model here.")
model_name = st.text_input("Model name", value="my cool model")
if st.button("Instantiate!"):
    base = os.path.join("models", model_name)
    os.makedirs(base, exist_ok=True)
    input_dict = os.path.join(base, "input_dict.json")
    with open(input_dict, "w") as f:
        json.dump(input_alphabet, f)
    output_dict = os.path.join(base, "output_dict.json")
    with open(output_dict, "w") as f:
        output_alphabet = set(st.session_state.output_vocab.tokens.values)
        output_alphabet = dict((w, i) for (i, w) in enumerate(output_alphabet))
        json.dump(output_alphabet, f)
    model = SimpleModel(
        d_input=len(input_alphabet),
        num_glosses=len(output_alphabet),
    )
    torch.save(model, os.path.join(base, "model.pt"))
    st.success("Initialized!")
