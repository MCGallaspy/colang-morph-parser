import json
import streamlit as st
import pandas as pd
import os
import torch
import tqdm
import re

from models import SimpleModel
from utils import encode_input, get_output, decode_output, get_entropy


class Evaluator:
    
    def __init__(self, model_dir):
        model_dict = torch.load(os.path.join("models", model_dir, "model.pt"))
        model = SimpleModel(model_dict['d_input'], model_dict['num_glosses'])
        model.load_state_dict(model_dict['weights'])
        
        base = os.path.join("models", model_dir)
        input_dict_fn = os.path.join(base, "input_dict.json")
        with open(input_dict_fn, "r") as f:
            self.input_dict = json.load(f)
        output_dict_fn = os.path.join(base, "output_dict.json")
        with open(output_dict_fn, "r") as f:
            self.output_alphabet = json.load(f)
        self.output_reversal = dict((i, k) for (k, i) in self.output_alphabet.items())
        self.model = model
        self.name = model_dir

    def get_preds(self, word):
        try:
            encoded = encode_input(word, self.input_dict)
            model_out = get_output(self.model, encoded, self.output_alphabet)
            entropy = get_entropy(model_out)
        except RuntimeError:
            entropy = 0
            model_out = None

        preds = decode_output(model_out, self.output_reversal)
        probs = torch.max(torch.exp(model_out), dim=1)[0][1:-1]
        return {"preds": preds, "entropy": entropy, "probs": probs}


os.makedirs("models", exist_ok=True)
model_dirs = os.listdir("models")
models = st.multiselect(
    "Choose a set of models to parse with:",
    model_dirs,
)

seps = {}
for model in models:
    sep = st.text_input(f"Output separator for {model}")
    seps[model] = sep

lang_text = st.text_area("Text to parse")

if st.button("Parse"):
    with st.spinner("Loading parsers"):
        parsers = [Evaluator(mdir) for mdir in models]
    parses = {}
    with st.spinner("Parsing..."):
        for word in re.split(r"\s+", lang_text):
            for parser in parsers:
                try:
                    parses[word] += [(parser.name, parser.get_preds(word))]
                except KeyError:
                    parses[word] = [(parser.name, parser.get_preds(word))]
    
    for word in parses.keys():
        st.markdown(f"## {word}")
        for parser, results in parses[word]:
            preds = results['preds']
            try:
                preds = preds[1:-1]
            except IndexError:
                preds = ['error']
            preds = seps[parser].join(preds)
            st.markdown(f"**{parser}**: {preds}")
            st.markdown(f"Mean probability: {torch.mean(results['probs']).item():.1%}", )
