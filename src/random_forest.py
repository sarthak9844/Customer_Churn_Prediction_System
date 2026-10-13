import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,f1_score,roc_auc_score,recall_score,classification_report,confusion_matrix)

from src.data_cleaning import load_data,clean_data
from src.feature_engineering import create_features,FeatureEngineer
from src.data_preprocessing import get_preprocessor

file_path = (r"D:\PROJECT\Customer_Churn_Prediction_System\data\Telco-Customer-Churn.csv")

df = load_data(file_path)
print(df.shape)

df = clean_data(df)
print(df.shape)

X = df.drop("Churn",axis=1)
y = df["Churn"].map({"No":0,"Yes":1})

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
print("X_train:",X_train.shape)
print("X_test:",X_test.shape)

X_train_for_preprocessor = create_features(X_train)
preprocessor = get_preprocessor(X_train_for_preprocessor)
pipeline = Pipeline([
    ("feature_engineering",FeatureEngineer()),
    ("preprocessor",preprocessor),
    ("classifier",RandomForestClassifier(n_estimators=200,random_state=42,max_depth=10,n_jobs=-1))
])

# using GridsearchCV for hyperparameter tunning
param_grid = {
            "classifier__n_estimators":[100,200],
            "classifier__max_depth":[5,10,None],
            "classifier__min_samples_split":[2,5],
            "classifier__min_samples_leaf":[1,2],
            "classifier__class_weight":[None,"balanced"] 
}

grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=5,
                scoring="f1",
                verbose=1,
                n_jobs=-1
)

grid_search.fit(X_train,y_train)

print("best parameter:",grid_search.best_params_)
print("best scoring:",grid_search.best_score_)
print("Training completed")

model = grid_search.best_estimator_

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

print(classification_report(y_test,y_pred,target_names=["No Churn", "Churn"]))
print(confusion_matrix(y_test, y_pred))

#save the model using the joblib
path =r"D:\PROJECT\Customer_Churn_Prediction_System\models\tuned_random_model.pkl"
joblib.dump(model,path)