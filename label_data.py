import json
import streamlit as st
import pandas as pd
import os

from models import SimpleModel

model_dirs = os.listdirs("models")
st.selectbox(
    "Choose a model to train:",
    model_dirs,
)

os.makedirs("datasets", exist_ok=True)
datasets = os.listdirs("datasets")
st.selectbox(
    "Choose a dataset:",
    model_dirs,
)

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