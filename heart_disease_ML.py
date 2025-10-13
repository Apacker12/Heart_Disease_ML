# Importing libraries
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier


# Defining default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# To read the cardio_train.csv data
df_original = pd.read_csv('cardio_train.csv', sep=';')
df_original.head()

# Removing any rows where the ap_lo or ap_hi is greater than 2000
heart_df = df_original[(df_original['ap_lo'] <= 2000) & (df_original['ap_hi'] <= 2000)]

# Creating a column where age is the number of years, rather than days
heart_df = heart_df.copy()
heart_df["age_years"] = (heart_df["age"] / 365.25).round()

# Creating BMI using height/weight
heart_df["bmi"] = (heart_df["weight"] / (heart_df["height"] / 100) ** 2).round(2)
heart_df.head()

# Separating the features from the target
X = heart_df.drop(columns=["cardio", "active", "smoke", "height", "alco", "id", "gender", "age_years"])
y = heart_df["cardio"].astype("int64")

# Separating the dataframe between training & test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline
forest_model = Pipeline([
    ("scaler", StandardScaler()),
    ("forest", RandomForestClassifier(random_state=42))])

# Fitting the pipeline to training data
forest_model.fit(x_train, y_train)

# Cross-validation via randomized search
forest_params = {
    "forest__n_estimators": [173, 174, 175],
    "forest__max_depth": [10, 11, 12],
    "forest__min_samples_split": [7, 8, 9],
    "forest__min_samples_leaf": [1],
    "forest__max_features": ["log2"]}

forest_grid = RandomizedSearchCV(
    estimator=forest_model,
    param_distributions=forest_params,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42)
forest_grid.fit(x_train, y_train)

# Creating the best model for the regressor
forest_best_model = forest_grid.best_estimator_

#Saving the model
joblib.dump(forest_best_model, "forest_best_model.pkl")
