# data_explore.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_explore():
    st.title("Data Exploration")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            df = pd.read_excel(uploaded_file)
        
        # Display first 10 rows of DataFrame
        st.title("Displaying First 10 Rows of DataFrame")
        st.write(df.head(10))
        
        # Display summary statistics
        st.title("Summary Statistics of DataFrame")
        st.write(df.describe())
        
        # Distribution of Age
        st.title("Distribution of Age")
        plt.figure(figsize=(10, 6))
        plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Distribution of Age')
        st.pyplot(plt)
        
        # Distribution of Sex
        st.title("Distribution of Sex")
        plt.figure(figsize=(10, 6))
        sex_counts = df['sex'].value_counts()
        sex_counts.plot(kind='bar', color='skyblue')
        plt.xlabel('Sex (0 = female, 1 = male)')
        plt.ylabel('Count')
        plt.title('Distribution of Sex')
        st.pyplot(plt)
        
        # Age vs Max Heart Rate
        st.title("Age vs Max Heart Rate")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['age'], df['thalach'], color='skyblue')
        ax.set_xlabel('Age')
        ax.set_ylabel('Max Heart Rate')
        ax.set_title('Age vs Max Heart Rate')
        st.pyplot(fig)
        
        # Correlation Matrix
        st.title("Correlation Matrix")
        st.write("Heatmap of the correlation matrix")
        corr_matrix = df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, linewidth=0.5, fmt='.2f', cmap='viridis_r')
        st.pyplot(plt)
        # Exploratory Data Analysis
        # st.title("Exploratory Data Analysis")
        
        # # Pairplot of Major Variables
        # st.subheader("Pairplot of Major Variables")
        # pairplot_fig = sns.pairplot(df, hue='target', diag_kind='kde')
        # st.pyplot(pairplot_fig)
        
        # FacetGrid: Age vs Max Heart Rate by Sex
        st.subheader("FacetGrid: Age vs Max Heart Rate by Sex")
        facet_grid_fig = sns.FacetGrid(df, col="sex", hue="target")
        facet_grid_fig.map(plt.scatter, "age", "thalach", alpha=0.7)
        facet_grid_fig.add_legend()
        st.pyplot(facet_grid_fig)
        
        # lmplot: Age vs Cholesterol
        st.subheader("lmplot: Age vs Cholesterol")
        lmplot_fig = sns.lmplot(x="age", y="chol", hue="target", data=df)
        st.pyplot(lmplot_fig)
        
        # Jointplot: Max Heart Rate vs ST Depression
        st.title("Jointplot: Max Heart Rate vs ST Depression")
        jointplot_fig = sns.jointplot(x="thalach", y="oldpeak", kind="hex", data=df)
        st.pyplot(jointplot_fig)
        
        
