import streamlit as st

pg = st.navigation([
    st.Page("initialize_dataset.py"),
    st.Page("initialize_model.py"),
    st.Page("label_data.py"),
    st.Page("train_model.py"),
    st.Page("evaluate_model.py"),
])

pg.run()