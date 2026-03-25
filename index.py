import streamlit as st

pg = st.navigation([
    st.Page("initialize_model.py"),
    st.Page("label_data.py"),
])

pg.run()