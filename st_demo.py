import streamlit as st
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import xgboost as xgb
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def load_dataset(name):
    if name == "Financial":
        # Replace with your dataset
        data = pd.read_csv("datasets/loan_approval_dataset.csv")
        data.columns = data.columns.str.strip()
        data = data.drop(columns=['loan_id'])
        # Remove leading/trailing spaces from the categorical column values
        data['education'] = data['education'].str.strip()
        data['self_employed'] = data['self_employed'].str.strip()
        data['loan_status'] = data['loan_status'].str.strip()
        # Encode categorical variables
        data['education'] = data['education'].map({'Graduate': 1, 'Not Graduate': 0})
        data['self_employed'] = data['self_employed'].map({'Yes': 1, 'No': 0})
        data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})

    elif name == "NLP":
        # Replace with your dataset and all the preprocessing steps
        data = pd.read_csv("datasets/nlp_dataset.csv")

    elif name == "Healthcare":
        # Replace with your dataset and all the preprocessing steps
        data = pd.read_csv("datastes/healthcare_dataset.csv")

    return data

def load_models(dataset_name):
    if dataset_name == "Financial":
        return joblib.load("models/loan_models.pkl")
    elif dataset_name == "NLP":
        return joblib.load("models/nlp_models.pkl")
    elif dataset_name == "Healthcare":
        return joblib.load("models/healthcare_models.pkl")
    

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Model Interpretability Visualization with LIME and SHAP")

    # Create different sections for each dataset
    st.subheader("1. Select a Dataset")
    dataset = st.selectbox("Choose a dataset:", ["Financial", "Healthcare", "NLP"])
     
    # Perform different interpretability methods on the first dataset
    if dataset == "Financial":
        # 1. Load the dataset
        X = load_dataset(dataset)
        st.write(f"{dataset} Dataset Sample")
        st.write(X.head())

        # 2. Select interpretability method
        st.subheader("2. Select an Interpretability Method")
        method = st.selectbox("Choose an interpretability method:", ["LIME", "SHAP"])

        if method == "SHAP":
            st.subheader("3. Interpretability using SHAP")
            # SHAP analysis
            loaded_models = load_models(dataset)
            model = loaded_models['XG Boost']
            sns.set_style('whitegrid')
            X = X.drop(columns=["loan_status"]).copy()
            X = X.astype(float)
            explainer = shap.Explainer(model)
            shap_values = explainer(X)

            # Visualize SHAP values
            idx = st.slider("Select Test Instance", 0, len(X) - 1, 0)
            st.write("SHAP Force Plot for a Single Prediction")
            shap.force_plot(explainer.expected_value, shap_values[idx].values, X.iloc[idx], matplotlib=True, show=False)
            st.pyplot(bbox_inches='tight')
            st.write("SHAP Summary Plot")
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(bbox_inches='tight')
            st.write("SHAP Bar Plot")
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')
            # force_plot = shap.force_plot(explainer.expected_value, shap_values[:1000], X.iloc[:1000])
            # shap.save_html("shap_force_plot.html", force_plot)
            # HtmlFile = open(f'shap_force_plot.html', 'r', encoding='utf-8')
            # components.html(HtmlFile.read(), height=600)

        elif method == "LIME":
            st.subheader("3. Interpretability using LIME")
            # Choose model type
            model_choice = st.radio("Select Model", ["Logistic Regression", 'Decision Tree', 'XG Boost', "Random Forest"])
            loaded_models = load_models(dataset)
            model = loaded_models[model_choice]
            sns.set_style('whitegrid')
            x = X.iloc[: , :-1].values
            y = X.iloc[: , -1].values
            X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=0.25,
                                                                random_state=42)
            target = ['Rejected', 'Approved']
            labels = {'0': 'Rejected', '1': 'Approved'}
            idx = st.slider("Select Test Instance", 0, len(X_test) - 1, 0)

            # Explain the prediction instance using LIME
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=list(X.columns),
                class_names=target,
                discretize_continuous=True,
                )
            exp = explainer.explain_instance(
                X_test[idx],
                model.predict_proba,
                )

            # Visualize the explanation
            st.write("LIME Explanation")
            exp.save_to_file('lime_explanation.html')
            HtmlFile = open(f'lime_explanation.html', 'r', encoding='utf-8')
            components.html(HtmlFile.read(), height=600)
            st.write('True label:', labels[str(y_test[idx])])
            st.write("Effect of Predictors")
            exp.as_pyplot_figure()
            st.pyplot(bbox_inches='tight')


    # Perform different interpretability methods on the second dataset
    elif dataset == "Healthcare":
        X = load_dataset(dataset)
        st.write(f"{dataset} Dataset Sample")
        st.write(X.head())

    # Perform different interpretability methods on the third dataset
    elif dataset == "NLP":
        X = load_dataset(dataset)
        st.write(f"{dataset} Dataset Sample")
        st.write(X.head())


if __name__ == "__main__":
    main()