import streamlit as st

st.set_page_config(layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = "HOME"

col1, col2, col3, col4, col5, col6 = st.columns([4, 1, 1, 1, 1, 1])

with col1:
    st.markdown("### 🏥 Alzheimer Detection")

with col2:
    if st.button("HOME", type="tertiary"): st.session_state.page = "HOME"
with col3:
    if st.button("ABOUT", type="tertiary"): st.session_state.page = "ABOUT"
with col4:
    if st.button("MODEL", type="tertiary"): st.session_state.page = "MODEL"
with col5:
    if st.button("ACCURACY", type="tertiary"): st.session_state.page = "ACCURACY"
with col6:
    st.button("Login", type="primary")

st.markdown(f"## Current Page: {st.session_state.page}")
