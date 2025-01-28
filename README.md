# DSC-261-Project

Our project focuses on addressing the challenge of distribution shifts in graph data, particularly in graph classification tasks. By leveraging invariant graph representations and causal inference techniques, we develop robust models capable of extrapolating across varying data distributions. This approach enhances generalization, improves explainability, and mitigates biases in graph neural networks.

## Setup

Ensure your environment is set up to run Python (ideally 3.9+) and Jupyter Notebooks. For installation instructions, please see this page: https://jupyter.org/install.

You will need to install OpenMP to run XGBoost. Refer to these instructions: https://xgboost.readthedocs.io/en/latest/install.html. If running on Mac OS, you will need to run `brew install libomp`.

To install the required Python packages needed to run the notebooks in this repository, run:
```
pip install -r requirements.txt
```

## Running the Notebooks
Simply run the cells in sequential order.

## Running the streamlit demo
Run the following command to open the app in your browser:
```
streamlit run st_demo_updated.py
```
