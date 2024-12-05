from imblearn.pipeline import Pipeline as imbPipeline
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
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as imbPipeline


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
        data = pd.read_csv("datasets/healthcare_dataset.csv")
        data.columns = data.columns.str.strip()
        data = data.drop_duplicates()
        data = data[data['gender'] != 'Other']
        def recategorize_smoking(smoking_status):
            if smoking_status in ['never', 'No Info']:
                return 0
            elif smoking_status == 'current':
                return 1
            elif smoking_status in ['ever', 'former', 'not current']:
                return 2

        data['smoking_history'] = data['smoking_history'].apply(recategorize_smoking)
        data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})


    return data

def load_models(dataset_name):
    if dataset_name == "Financial":
        return joblib.load("models/loan_models.pkl")
    elif dataset_name == "NLP":
        return joblib.load("models/nlp_models.pkl")
    elif dataset_name == "Healthcare":
        model_path = "models/healthcare_models.pkl"
        model = joblib.load(model_path)
        return {"Random Forest": model}
    

def main():
    plt.style.use('default')
    st.title("Model Interpretability Visualization with LIME and SHAP")

    # Create different sections for each dataset
    st.subheader("1. Select a Dataset")
    dataset = st.selectbox("Choose a dataset:", ["Financial", "Healthcare"])
     
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
        data = load_dataset(dataset)
        st.write(f"{dataset} Dataset Sample")
        st.write(data.head())

        st.subheader("2. Select an Interpretability Method")
        method = st.selectbox("Choose an interpretability method:", ["LIME", "SHAP"])

        loaded_models = load_models(dataset)
        model = loaded_models.get('Random Forest')

        idx = st.slider("Select Test Instance", 0, 24031, 0)  


        if method == "SHAP":
            st.subheader("3. Interpretability using SHAP")
            loaded_models = load_models(dataset)
            model = loaded_models.get('Random Forest')
            if model and isinstance(model, imbPipeline):
                st.write("Model loaded and is a valid pipeline.")
                try:
                    if 'classifier' in model.named_steps:
                        tree_model = model.named_steps['classifier']
                        if isinstance(tree_model, RandomForestClassifier):
                            explainer = shap.TreeExplainer(tree_model)
                            X_shap = data.drop(columns=["diabetes"])
                            st.write(f"Data shape for SHAP: {X_shap.shape}")

                            sample_size = 1000
                            X_sample = X_shap.sample(n=sample_size, random_state=42)
                            st.write(f"Using a sample of {sample_size} instances for SHAP analysis.")

                            shap_values = explainer.shap_values(X_sample)

                            st.write(f"SHAP values shape: {np.array(shap_values).shape}")

                            idx = st.slider("Select Test Instance", 0, len(X_sample) - 1, 0)
                            st.write("SHAP Force Plot for a Single Prediction")
                            shap.force_plot(explainer.expected_value[1], shap_values[1][idx, :], X_sample.iloc[idx, :], matplotlib=True, show=False)
                            st.pyplot(bbox_inches='tight')

                            st.write("SHAP Summary Plot")
                            shap.summary_plot(shap_values[1], X_sample, show=False)
                            st.pyplot(bbox_inches='tight')

                            st.write("SHAP Bar Plot")
                            shap.summary_plot(shap_values[1], X_sample, plot_type="bar", show=False)
                            st.pyplot(bbox_inches='tight')
                        else:
                            st.error("The classifier in the pipeline is not a RandomForest.")
                    else:
                        st.error("RandomForest classifier not found in the pipeline.")
                except Exception as e:
                    st.error(f"Error during SHAP analysis: {e}")
            else:
                st.error("Model could not be loaded or is not a valid RandomForest pipeline.")


        elif method == "LIME":
            st.subheader("3. Interpretability using LIME")
            model_choice = st.radio("Select Model", ["Random Forest"])
            model = loaded_models.get('Random Forest')
            sns.set_style('whitegrid')
            x = data.drop(columns=["diabetes"]) 
            y = data["diabetes"]
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

            target = ['Non-Diabetic', 'Diabetic']

            explainer = LimeTabularExplainer(
                X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=target,
                verbose=True,
                mode='classification'
                )

            instance = X_test.iloc[idx].values.reshape(1, -1)  

            def model_predict(instance):
                return model.predict_proba(pd.DataFrame(instance, columns=X_train.columns))

            exp = explainer.explain_instance(
                data_row=instance[0], 
                predict_fn=model_predict
                )

            st.write("LIME Explanation")
            exp.save_to_file('lime_explanation.html')
            HtmlFile = open('lime_explanation.html', 'r', encoding='utf-8')
            components.html(HtmlFile.read(), height=600)
            st.write('True label:', target[y_test.iloc[idx]])  
            st.write("Effect of Predictors")
            exp.as_pyplot_figure()
            st.pyplot(bbox_inches='tight')




if __name__ == "__main__":
    main()