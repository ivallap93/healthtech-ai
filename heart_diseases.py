"""
EDA on Heart Disease Dataset by UCL
Fully functional exploratory data analysis notebook
"""

# %%
# 1. Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Optional visualization / profiling
from pandas_profiling import ProfileReport
import dtale
import klib

sns.set(style="whitegrid")

print("Libraries imported successfully")

# %%
# 2. Fetch and load the heart disease dataset (UCI dataset id=45)
heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data['features']   # shape (303, 13)
y = heart_disease.data['targets']    # shape (303,)
Ids = heart_disease.data['ids']      # shape (303,)

# Create DataFrame
df = pd.DataFrame(X)
df['target'] = y
print(df.head())
print(df.info())

# %%
# 3. Basic statistics
print(df.describe().round(2))

# %%
# 4. Check missing values
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)
missing_cols = df.columns[df.isnull().any()]
print("Columns with missing values:", missing_cols.tolist())
# Impute missing values if any (median for numeric)
for col in missing_cols:
    df[col].fillna(df[col].median(), inplace=True)
# %%
# 5. Target distribution
sns.countplot(x='target', data=df)
plt.title("Target Class Distribution")
plt.show()
print(df['target'].value_counts(normalize=True))

# %%
# 6. Manual Univariate Analysis
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove('target')  # exclude target

# Histograms & KDE plots
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# %%
# 7. Boxplots and Outlier Detection
def identify_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    STD = df[column].std()
    lower_bound = Q1 - 2*STD
    upper_bound = Q3 + 2*STD
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Outliers in {column} ({len(outliers)} rows):")
    print(outliers[[column]])
    # Plot boxplot
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()


for col in numeric_cols:
    identify_outliers(df, col)

# %%
# 8. Categorical Feature Analysis (if any categorical columns exist)
categorical_cols = df.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    print(f"Value counts for {col}:")
    print(df[col].value_counts())
    sns.countplot(data=df, x=col)
    plt.title(f"Count plot of {col}")
    plt.show()

# %%
# 9. Bivariate Analysis: Numeric vs Target
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='target', y=col, data=df)
    plt.title(f'{col} vs Target')
    plt.show()

# %%
# 10. Bivariate Analysis: Categorical vs Target
for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue='target', data=df)
    plt.title(f'{col} vs Target')
    plt.show()

# %%
# 11. Correlation Analysis
corr_matrix = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Optional: Pairplot for numeric features
sns.pairplot(df, hue='target', vars=numeric_cols)
plt.show()

# %%
# 12. Summary of EDA
print("EDA Summary:")
print(f"- Number of rows: {df.shape[0]}")
print(f"- Number of columns: {df.shape[1]}")
print(f"- Numeric features: {numeric_cols}")
print(f"- Categorical features: {categorical_cols}")
print(f"- Columns with missing values: {missing_cols.tolist()}")
print("Target class distribution:\n", df['target'].value_counts())
#%%
# Scaling numeric features
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

# %%
# 11. Train/Test Split
X = df_scaled.drop(['target'], axis=1)
y = df_scaled['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
# 12. Baseline Models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Feature importance for RandomForest
    if name == 'RandomForest':
        importances = model.feature_importances_
        feat_importance = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        plt.figure(figsize=(10,6))
        sns.barplot(x=feat_importance, y=feat_importance.index)
        plt.title("RandomForest Feature Importances")
        plt.show()



# %%
# 13. Optional automated profiling
from pandas_profiling import ProfileReport

profile = ProfileReport(df, title="Heart Disease Data Profile", explorative=True)
profile.to_file("heart_disease_profile.html")
print("Pandas profiling report saved as 'heart_disease_profile.html'")

# %%
# 14. Optional: Interactive data exploration (dtale)
import dtale
dtale.show(df)

# %%
# 15. Optional: Klib visualization & cleaning
import klib
klib.cat_plot(df)
klib.corr_mat(df)
klib.corr_plot(df)
klib.dist_plot(df)
klib.missingval_plot(df)
klib.data_cleaning(df)

# %%
