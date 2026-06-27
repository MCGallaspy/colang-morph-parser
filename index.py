import streamlit as st

pg = st.navigation([
    st.Page("initialize_dataset.py", title="initialize word lists"),
    st.Page("initialize_model.py"),
    st.Page("label_data.py", title="label word lists"),
    st.Page("train_model.py"),
    st.Page("evaluate_model.py"),
    st.Page("parser.py"),
])

pg.run()