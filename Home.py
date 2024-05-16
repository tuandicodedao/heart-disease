# Home.py
import streamlit as st
from utilities import apply_style, read_markdown_file
import data_explore
import model_training
from predict import predict
#st.set_page_config(page_title="Heart Disease Prediction App", layout="wide")
baseURL = 'https://heart_disease0710.streamlit.app/'
def main():
    apply_style("style.css")
    
    # st.title("Streamlit Dashboard")
    st.sidebar.title("Navigation")
    
    menu = ["Home", "Data Exploration", "Model Training","Predict"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Welcome to the Streamlit Dashboard")
        st.markdown(read_markdown_file("README.md"))
    elif choice == "Data Exploration":
        data_explore.data_explore()
    elif choice == "Model Training":
        model_training.model_training()
    elif choice == "Predict":
        predict()

if __name__ == '__main__':
    main()
