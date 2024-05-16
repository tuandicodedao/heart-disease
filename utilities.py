# utilities.py
import streamlit as st

def apply_style(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def read_markdown_file(markdown_file):
    return open(markdown_file, 'r').read()
