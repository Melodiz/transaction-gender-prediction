# Gender Prediction Model

## Project Overview
The Gender Prediction Model project aims to develop a machine learning model that can accurately predict the gender of a client using their transaction data. The project includes data preparation and analysis, model tuning, and evaluation metrics.

## Getting Started
To start working on this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone git@github.com:Melodiz/transaction-gender-prediction.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Gender_transaction_base
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4. Download the [data](https://www.kaggle.com/datasets/okunevda/hsexsber) from Kaggle
5. Leave the unpacked data in a folder named `data` in the root of the repository.

## Project Structure
The project's directory structure is as follows:

```
Gender_transaction_base/
├── LICENSE
├── README.md
├── gender_by_transaction.ipynb
├── requirements.txt
└── data/
    ├── train.csv
    ├── test.csv
    ├── mcc_codes.csv
    ├── transactions.csv
    ├── trans_types.csv
    └── test_sample_submission.csv
```
## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- catboost
- scikit-learn
- xgboost
- lightgbm
- nltk
- gensim
- @jupyter-widgets/base
- jquery
- lodash
- plotly.js-dist-min
