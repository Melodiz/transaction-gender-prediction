import pandas as pd
import numpy as np
import plotly.express as px
import re
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_SUB_DIR = '/Users/melodiz/projects/Gender_transaction_base/data/'

# Load data
print("Loading data...")
df_transactions = pd.read_csv(DATA_SUB_DIR + 'transactions.csv', sep=',')
df_mcc_codes = pd.read_csv(DATA_SUB_DIR + 'mcc_codes.csv', sep=';')
df_trans_types = pd.read_csv(DATA_SUB_DIR + 'trans_types.csv', sep=';')
df_train = pd.read_csv(DATA_SUB_DIR + 'train.csv', sep=',', index_col='client_id').drop('Unnamed: 0', axis=1)

# Display info about transactions data
df_transactions.info()

# Define a range of intervals to evaluate
intervals = [(-x*3_000, x*3_000) for x in range(13, 20)]

# Function to evaluate outliers for a given interval
def evaluate_outliers(df, interval):
    lower_bound, upper_bound = interval
    outliers = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]
    num_outliers = len(outliers)
    percentage_outliers = num_outliers / len(df) * 100
    return num_outliers, percentage_outliers

# Evaluate each interval
print("Evaluating outliers...")
results = []
for interval in intervals:
    num_outliers, percentage_outliers = evaluate_outliers(df_transactions, interval)
    results.append({
        'interval': interval,
        'num_outliers': num_outliers,
        'percentage_outliers': round(percentage_outliers, 3)
    })

# Convert results to DataFrame for easy visualization
df_results = pd.DataFrame(results)

# Print results
print(df_results)

# Clear memory from new temporary variables
del num_outliers, percentage_outliers, results, intervals, df_results, interval

# It looks like the interval (-50_000, 50_000) is a good compromise for identifying outliers in the 'amount' column.
# So drop the outliers outside of this interval from the transactions dataframe.
print("Dropping outliers...")
df_transactions = df_transactions[(df_transactions['amount'] >= -50_000) & (df_transactions['amount'] <= 50_000)]

# Format the time of transaction
print("Formatting transaction time...")
df_transactions['day'] = df_transactions['trans_time'].str.split().apply(lambda x: int(x[0]) % 7)
df_transactions['hour'] = df_transactions['trans_time'].apply(lambda x: int(x.split()[1].split(':')[0]))
df_transactions['is_night'] = df_transactions['hour'].apply(lambda x: 1 if x >= 22 or x <= 6 else 0)

# Drop term_id since it's not informative for this analysis
df_transactions.drop('term_id', axis=1, inplace=True)

# One-hot encode the mcc_code and trans_type columns
print("One-hot encoding categorical variables...")
df_transactions = pd.get_dummies(df_transactions, columns=['mcc_code', 'trans_type'])

# Scaling and Normalization
print("Scaling numerical features...")
scaler = StandardScaler()
numerical_features = ['amount']
df_transactions[numerical_features] = scaler.fit_transform(df_transactions[numerical_features])

# Aggregated Features: Aggregate transaction data to create summary statistics for each client
print("Aggregating transaction data...")
agg_features = df_transactions.groupby('client_id').agg({
    'amount': ['sum', 'mean', 'std', 'min', 'max'],
    'day': ['nunique'],
    'hour': ['nunique'],
    'is_night': ['sum']
})
agg_features.columns = ['_'.join(col).strip() for col in agg_features.columns.values]
df_train = df_train.join(agg_features, on='client_id')

# Handle NaN values by filling them with the median of the respective columns
print("Handling NaN values...")
df_train.fillna(df_train.median(), inplace=True)

# Correlation Analysis: Remove highly correlated features to reduce multicollinearity
print("Performing correlation analysis...")
corr_matrix = df_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df_train.drop(to_drop, axis=1, inplace=True)

# Handling Imbalanced Data using SMOTE
print("Handling imbalanced data using SMOTE...")
X = df_train.drop('gender', axis=1)
y = df_train['gender']
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print the shape of the resampled data
print("Resampled X shape:", X_resampled.shape)
print("Resampled y shape:", y_resampled.shape)

# Train-Test Split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    print(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    print(f"Making predictions with {model.__class__.__name__}...")
    y_pred = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
    print("\n")

# Hyperparameter tuning for RandomForestClassifier
print("Tuning RandomForestClassifier...")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_
train_and_evaluate_model(best_rf_model, X_train, y_train, X_test, y_test)

# Hyperparameter tuning for GradientBoostingClassifier
print("Tuning GradientBoostingClassifier...")
gbt_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
gbt_grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), gbt_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
gbt_grid_search.fit(X_train, y_train)
best_gbt_model = gbt_grid_search.best_estimator_
train_and_evaluate_model(best_gbt_model, X_train, y_train, X_test, y_test)

# Hyperparameter tuning for SVC
print("Tuning SVC...")
svc_param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}
# svc_grid_search = GridSearchCV(SVC(random_state=42, probability=True), svc_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
# svc_grid_search.fit(X_train, y_train)
# best_svc_model = svc_grid_search.best_estimator_
# train_and_evaluate_model(best_svc_model, X_train, y_train, X_test, y_test)

# Hyperparameter tuning for CatBoostClassifier
print("Tuning CatBoostClassifier...")
catboost_param_grid = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [3, 5, 7]
}
catboost_grid_search = GridSearchCV(CatBoostClassifier(random_state=42, verbose=0), catboost_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
catboost_grid_search.fit(X_train, y_train)
best_catboost_model = catboost_grid_search.best_estimator_
train_and_evaluate_model(best_catboost_model, X_train, y_train, X_test, y_test)