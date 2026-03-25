import streamlit as st
import pandas as pd

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
    st.write(st.session_state.output_vocab)

output_tokens = set(st.session_state.output_vocab.tokens)

st.markdown("**Inflectional element builder**")

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