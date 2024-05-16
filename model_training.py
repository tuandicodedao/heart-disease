import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def model_training():
    st.title("Model Training")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            df = pd.read_excel(uploaded_file)

        st.write(df.head())

        target = st.selectbox("Select the target column", df.columns)
        feature_columns = st.multiselect("Select feature columns", df.columns, default=[col for col in df.columns if col != target])

        X = df[feature_columns]
        y = df[target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the models and parameters for GridSearchCV
        models = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=100),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear']
                }
            }
        }

        # Fit and score the models
        def fit_and_score(models, X_train, X_test, y_train, y_test):
            np.random.seed(42)
            model_scores = {}
            best_estimators = {}
            performance_metrics = {}

            for model_name, model_data in models.items():
                clf = GridSearchCV(model_data['model'], model_data['params'], cv=3, verbose=False, n_jobs=-1)
                clf.fit(X_train, y_train)
                best_estimators[model_name] = clf.best_estimator_
                y_pred = clf.predict(X_test)
                model_scores[model_name] = accuracy_score(y_test, y_pred)
                
                # Calculate performance metrics
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)

                performance_metrics[model_name] = {
                    'accuracy': model_scores[model_name],
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm
                }

            return performance_metrics, best_estimators

        if st.button("Train Model"):
            performance_metrics, best_estimators = fit_and_score(models, X_train, X_test, y_train, y_test)

            for model_name, metrics in performance_metrics.items():
                st.write(f"### {model_name}")
                st.write("**Accuracy**:", metrics['accuracy'])
                st.write("**Precision**:", metrics['precision'])
                st.write("**Recall**:", metrics['recall'])
                st.write("**F1 Score**:", metrics['f1_score'])

                fig, ax = plt.subplots()
                cm_display = ConfusionMatrixDisplay(confusion_matrix=metrics['confusion_matrix'], display_labels=best_estimators)
                
                
                # Plot ROC Curve
                y_prob = best_estimators[model_name].predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                st.title("Receiver Operating Characteristic (ROC) Curve")
                st.write("Area under the curve (AUC):", roc_auc)

                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                st.pyplot(plt)
                # Hiển thị feature importance
                model_coefs = best_estimators[model_name].coef_
                feature_dict = dict(zip(X_train.columns, model_coefs[0]))
                feature_df = pd.DataFrame(feature_dict, index=[0])
                st.title('Feature Importance of Logistic Regression')
                st.bar_chart(feature_df.T)