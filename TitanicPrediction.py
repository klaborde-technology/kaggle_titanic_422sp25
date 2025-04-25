# Author: Keaton LaBorde & Michael Maxwell
# Class: CSCI 422

# https://www.youtube.com/watch?v=6IGx7ZZdS74

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
from warnings import simplefilter
import warnings

simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*convergence.*")


test_df = pd.read_csv("test.csv")

train_df = pd.read_csv("train.csv")

# 5 values for ints, 4 for objects, 2 for floats
# print(train_df.describe(include=["O"]))

# print(train_df.groupby(["Pclass"], as_index=False)["Survived"].mean())

# print(train_df.groupby(["SibSp"], as_index=False)["Survived"].mean())

# print(train_df.groupby(["Parch"], as_index=False)["Survived"].mean())

train_df["Family_Size"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["Family_Size"] = test_df["SibSp"] + test_df["Parch"] + 1

# print(train_df.groupby(["Family_Size"], as_index=False)["Survived"].mean())

family_map = {1: "Alone", 2: "Small", 3: "Small", 4: "Small", 5: "Medium", 6: "Medium", 7: "Large", 8: "Large", 11: "Large"}

train_df["Family_Size_Grouped"] = train_df["Family_Size"].map(family_map)
test_df["Family_Size_Grouped"] = test_df["Family_Size"].map(family_map)

# print(train_df.groupby(["Family_Size_Grouped"], as_index=False)["Survived"].mean())

train_df["Age_Cut"] = pd.qcut(train_df["Age"], 8)
test_df["Age_Cut"] = pd.qcut(test_df["Age"], 8)

# print(train_df.groupby(["Age_Cut"], as_index=False)["Survived"].mean())

train_df["Fare_Cut"] = pd.qcut(train_df["Fare"], 6)
test_df["Fare_Cut"] = pd.qcut(test_df["Fare"], 6)

# print(train_df.groupby(["Fare_Cut"], as_index=False)["Survived"].mean())

train_df["Title"] = train_df["Name"].str.split(pat = ",", expand=True)[1].str.split(pat=".", expand=True)[0].apply(lambda x: x.strip())
test_df["Title"] = test_df["Name"].str.split(pat = ",", expand=True)[1].str.split(pat=".", expand=True)[0].apply(lambda x: x.strip())

# print(train_df.groupby(["Sex"], as_index=False)["Survived"].mean())

train_df["Title"] = train_df["Title"].replace({
    "Capt": "Military",
    "Col": "Military",
    "Major": "Military",
    "Jonkheer": "Noble",
    "the Countess": "Noble",
    "Don": "Noble",
    "Lady": "Noble",
    "Sir": "Noble",
    "Mlle": "Noble",
    "Ms": "Noble",
    "Mme": "Noble"
    
})

test_df["Title"] = test_df["Title"].replace({
    "Capt": "Military",
    "Col": "Military",
    "Major": "Military",
    "Jonkheer": "Noble",
    "the Countess": "Noble",
    "Don": "Noble",
    "Lady": "Noble",
    "Sir": "Noble",
    "Mlle": "Noble",
    "Ms": "Noble",
    "Mme": "Noble"
    
})

# print(train_df.groupby(["Title"], as_index=False)["Survived"].mean())

train_df["TicketNumber"] = train_df["Ticket"].apply(lambda x: pd.Series({"Ticket": x.split()[-1]}))
test_df["TicketNumber"] = test_df["Ticket"].apply(lambda x: pd.Series({"Ticket": x.split()[-1]}))

# print(train_df.groupby(["TicketNumber"], as_index=False)["Survived"].mean())

# print(train_df.groupby("TicketNumber")["TicketNumber"].transform("count"))

train_df["TicketNumberCounts"] = train_df.groupby("TicketNumber")["TicketNumber"].transform("count")
test_df["TicketNumberCounts"] = test_df.groupby("TicketNumber")["TicketNumber"].transform("count")

# print(train_df.groupby(["TicketNumberCounts"], as_index=False)["Survived"].agg(["count", "mean"]))

# print(train_df["Ticket"].str.split(pat=" ", expand=True))

train_df["TicketLocation"] = np.where(train_df["Ticket"].str.split(pat=" ", expand=True)[1].notna(), train_df["Ticket"].str.split(pat=" ", expand=True)[0].apply(lambda x: x.strip()), "Blank")
test_df["TicketLocation"] = np.where(test_df["Ticket"].str.split(pat=" ", expand=True)[1].notna(), test_df["Ticket"].str.split(pat=" ", expand=True)[0].apply(lambda x: x.strip()), "Blank")

# print(train_df["TicketLocation"].value_counts())

train_df["TicketLocation"] = train_df["TicketLocation"].replace({
    "SOTON/O.Q.": "SOTON/OQ",
    "C.A.": "CA",
    "SC/PARIS": "SC/Paris",
    "S.C./PARIS": "SC/Paris",
    "A/4.": "A/4",
    "A/5.": "A/5",
    "A./5.": "A/5",
    "W./C.": "W/C"
    })

test_df["TicketLocation"] = test_df["TicketLocation"].replace({
    "SOTON/O.Q.": "SOTON/OQ",
    "C.A.": "CA",
    "SC/PARIS": "SC/Paris",
    "S.C./PARIS": "SC/Paris",
    "A/4.": "A/4",
    "A/5.": "A/5",
    "A./5.": "A/5",
    "W./C.": "W/C"
    })

# print(train_df.groupby(["TicketLocation"], as_index=False)["Survived"].mean())

train_df["Cabin"] = train_df["Cabin"].fillna("U")
train_df["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else "x" for i in train_df["Cabin"]])
test_df["Cabin"] = test_df["Cabin"].fillna("U")
test_df["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else "x" for i in test_df["Cabin"]])

# print(train_df.groupby(["Cabin"], as_index=False)["Survived"].agg(["count", "mean"]))

train_df["Cabin_Assigned"] = train_df["Cabin"].apply(lambda x: 0 if x in ["U"] else 1)
test_df["Cabin_Assigned"] = test_df["Cabin"].apply(lambda x: 0 if x in ["U"] else 1)
# print(train_df.groupby(["Cabin_Assigned"], as_index=False)["Survived"].agg(["count", "mean"]))

train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)
train_df["Fare"].fillna(train_df["Fare"].mean(), inplace=True)

test_df["Age"].fillna(test_df["Age"].mean(), inplace=True)
test_df["Fare"].fillna(test_df["Fare"].mean(), inplace=True)

ohe = OneHotEncoder(sparse_output=False)
ode = OrdinalEncoder()
SI = SimpleImputer(strategy="most_frequent")

ode_cols = ["Family_Size_Grouped"]
ohe_cols = ["Sex", "Embarked"]

X = train_df.drop(["Survived"], axis = 1)
y = train_df["Survived"]
X_test = test_df.drop(["Age_Cut", "Fare_Cut"], axis = 1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

from sklearn.preprocessing import StandardScaler

scale_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="mean")),
    ("scale", StandardScaler())
])

ordinal_pipeline = Pipeline(steps = [
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

ohe_pipeline = Pipeline(steps = [
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

col_trans = ColumnTransformer(transformers=[
    ("ord_pipeline", ordinal_pipeline, ode_cols),
    ("ohe_pipeline", ohe_pipeline, ohe_cols),
    ("scale_pipeline", scale_pipeline, ["Age", "Fare", "TicketNumberCounts"]),
    ("passthrough", "passthrough", ["Pclass", "Cabin_Assigned"])
    ],
    remainder="drop",
    n_jobs=-1)

rfc = RandomForestClassifier()

param_grid = {
    "n_estimators": [100, 150, 200],
    "min_samples_split": [5, 10, 15],
    "max_depth": [8, 9, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
    }

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

pipefinalrfc = make_pipeline(col_trans, CV_rfc)
pipefinalrfc.fit(X_train, y_train)

print("------------------------RandomForestClassifier------------------------------------\n")
print(CV_rfc.best_params_)
print(CV_rfc.best_score_)

dtc = DecisionTreeClassifier()
param_grid = {
    "min_samples_split": [5, 10, 15],
    "max_depth": [10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
    }

CV_dtc = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

pipefinaldtc = make_pipeline(col_trans, CV_dtc)
pipefinaldtc.fit(X_train, y_train)

print("------------------------DecisionTreeClassifier------------------------------------\n")
print(CV_dtc.best_params_)
print(CV_dtc.best_score_)

knn = KNeighborsClassifier()

param_grid = {
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "p": [1, 2]
    }

CV_knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

pipefinalknn = make_pipeline(col_trans, CV_knn)
pipefinalknn.fit(X_train, y_train)

print("------------------------KNeighborsClassifier------------------------------------\n")
print(CV_knn.best_params_)
print(CV_knn.best_score_)

svc = SVC()

param_grid = {
    "C": [100, 10, 1.0, 0.1, 0.01, 0.001],
    "kernel": ["linear", "poly", "rbf", "sigmoid"]
    }

CV_svc = GridSearchCV(estimator=svc, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

pipefinalsvc = make_pipeline(col_trans, CV_svc)
pipefinalsvc.fit(X_train, y_train)

print("------------------------SVCClassifier------------------------------------\n")
print(CV_svc.best_params_)
print(CV_svc.best_score_)

lr = LogisticRegression(solver='saga', max_iter=500)


param_grid = {
    "C": [100, 10, 1.0, 0.1, 0.01, 0.001]
    }

CV_lr = GridSearchCV(estimator=lr, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

pipefinallr = make_pipeline(col_trans, CV_lr)
pipefinallr.fit(X_train, y_train)

print("------------------------LogisticRegression------------------------------------\n")
print(CV_lr.best_params_)
print(CV_lr.best_score_)

gnb = GaussianNB()

param_grid = {
    "var_smoothing": [0.00000001, 0.000000001, 0.00000001]
    }


CV_gnb = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))

pipefinalgnb = make_pipeline(col_trans, CV_gnb)
pipefinalgnb.fit(X_train, y_train)

print("------------------------GaussianNB------------------------------------\n")
print(CV_gnb.best_params_)
print(CV_gnb.best_score_)

y_pred = pipefinalrfc.predict(X_test)
y_pred_2 = pipefinaldtc.predict(X_test)
y_pred_3 = pipefinalknn.predict(X_test)
y_pred_4 = pipefinalsvc.predict(X_test)
y_pred_5 = pipefinallr.predict(X_test)
y_pred_6 = pipefinalgnb.predict(X_test)

submission1 = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": y_pred
    })

submission2 = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": y_pred_2
    })

submission3 = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": y_pred_3
    })

submission4 = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": y_pred_4
    })

submission5 = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": y_pred_5
    })

submission6 = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": y_pred_6
    })

submission1.to_csv("C:/Users/klabo/Data Mining/TitanicPrediction/RandomForestClassifer.csv", index=False)
submission2.to_csv("C:/Users/klabo/Data Mining/TitanicPrediction/DecisionTreeClassifier.csv", index=False)
submission3.to_csv("C:/Users/klabo/Data Mining/TitanicPrediction/KNeighborsClassifier.csv", index=False)
submission4.to_csv("C:/Users/klabo/Data Mining/TitanicPrediction/SVC.csv", index=False)
submission5.to_csv("C:/Users/klabo/Data Mining/TitanicPrediction/LogisticRegression.csv", index=False)
submission6.to_csv("C:/Users/klabo/Data Mining/TitanicPrediction/GaussianNB.csv", index=False)







