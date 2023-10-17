import argparse

from pathlib import Path

import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import mlflow
import mlflow.sklearn




def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to data")
    parser.add_argument("--model_output", type=str, help="Path to model")

    parser.add_argument("--base_estimator", type=object, default = DecisionTreeClassifier())
   
    parser.add_argument('--n_estimators', type=int, default = 10000,
                        help='Number of weak learners')

    parser.add_argument('--learning_rate', type=float, default = 1.0,
                        help='learning_rate')

    args = parser.parse_args()

    return args


def main(args):
    '''Read train dataset, train model, save trained model'''

    mlflow.sklearn.autolog()
    mlflow.log_param("adaboost_model", "AdaBoostClassifier")
    
    mlflow.log_param('base_estimator', object(args.estimator))
    mlflow.log_param('n_estimators', int(args.n_estimators))
    mlflow.log_param('learning_rate', float(args.learning_rate))
   
    
    print("reading data")  
    df = pd.read_csv(Path(args.input_data) / "data.csv", sep=";")

    Numeric_Var = ["List of numerical variables in your data frame"]
    Cat_Var = ["List of categorical variables in your data frame"]
    
   
    print("encoding variables")
    label_encoder = LabelEncoder()
    for column in Cat_Var:
        df[column] = label_encoder.fit_transform(df[column])

    df[Cat_Var] = df[Cat_Var].astype('category')

    print("normalisation des variables continues")    
    scaler = StandardScaler()
    #scaler = MaxAbsScaler()
    df[Numeric_Var] = scaler.fit_transform(df[Numeric_Var])
    
    print("splitting")
    X = df.drop("fraud", axis=1)
    Y = df[["fraud"]]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=15,shuffle=True,stratify=Y)  

    print("Y_train:\n",  Y_train.value_counts(), 
      "\nY_test:\n", Y_test.value_counts())
    
    sampling_strategy = {0: 11710063, 1:4687331}
    ros = RandomOverSampler(random_state=0, sampling_strategy=sampling_strategy)
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train)

    print("Class distribution after under-sampling:")
    print("X_train_resampled shape:", X_train_resampled.shape)
    print("Y_train_resampled shape:", Y_train_resampled.shape)
    print("Y_train_Count:", Y_train_resampled.value_counts())
    print(sorted(Counter(Y_train_resampled).items()))
    
    adaboost_model = AdaBoostClassifier(n_estimators= args.n_estimators,
                                                #base_estimator = base_classifier,
                                                learning_rate = args.learning_rate,
                                            )
    print("training model")                                            
    adaboost_model = adaboost_model.fit(X_train_resampled, Y_train_resampled)
    print("predicting values")
    y_adaboost = adaboost_model.predict(X_test)
    print("evaluating model")
    
    accuracyscore = metrics.accuracy_score(Y_test,y_adaboost)*100
    f1score = metrics.f1_score(Y_test,y_adaboost)*100
    recallscore = metrics.recall_score(Y_test,y_adaboost)*100
    specificityscore = metrics.recall_score(Y_test, y_adaboost, pos_label=0)*100
    precisionscore = metrics.precision_score(Y_test,y_adaboost)*100


    mlflow.log_metric("accuracy_score", accuracyscore)
    mlflow.log_metric("f1_score", f1score)
    mlflow.log_metric("precision_score", precisionscore)
    mlflow.log_metric("recall_score", recallscore)
    mlflow.log_metric("recall_score", specificityscore)
    mlflow.log_metric("specificity_score", specificityscore)
    print ("Accuracy=", accuracyscore)
    print ("f1.score=", f1score)
    print ("precision.score=", precisionscore)
    print ("recall_score=", recallscore)
    print("Specificity=", specificityscore)

    # Visualize results
    cm = confusion_matrix(Y_test, y_adaboost)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    cm_display.figure_.savefig('confusion_matrix.png')
    mlflow.log_artifact("confusion_matrix.png")

    ######## COURBE DE ROC
    # calculate the fpr and tpr for all thresholds of the classification
    prob = adaboost_model.predict_proba(X_test)
    pred = prob[:,1]
    fpr_d, tpr_d, threshold = metrics.roc_curve(Y_test, pred)
    roc_auc = metrics.auc(fpr_d, tpr_d)
    print("AUC Adaboost=", roc_auc)
    auc_d = roc_auc_score(Y_test, pred)
    plt.title('Courbe de ROC - Adaboost')
    plt.plot(fpr_d, tpr_d, 'b', label = 'AUC = %0.2f' % auc_d)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate - Sensitivity')
    plt.xlabel('False Positive Rate - (1-Specificity)')
    plt.savefig('roc_curve.png')
    mlflow.log_artifact("roc_curve.png")

    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(Y_test, adaboost_model.predict_proba(X_test)[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.savefig('precision_recall.png')
    mlflow.log_artifact("precision_recall.png")

    feature_importance = adaboost_model.feature_importances_
    print("Importance des caractéristiques :")
    importance_df = pd.DataFrame({'Caractéristiques': X_train_resampled.columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)

    

    feature_importances = adaboost_model.feature_importances_

    feature_names = X_train_resampled.columns

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Importance des Variables')
    plt.xlabel('Importance')
    plt.ylabel('Variable')
    plt.savefig('Importance des Variables.png')
    mlflow.log_artifact("Importance des Variables.png")

    registered_model_name="sklearn-Adaboost-model"

    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=adaboost_model,
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name
    )

    #Saving the model to a file
    print("Saving the model via MLFlow")
    mlflow.sklearn.save_model(
        sk_model=adaboost_model,
        path=os.path.join(registered_model_name, "Adaboost_model"),
    )



if __name__ == "__main__":
    
    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"input_data: {args.input_data}",
        f"model_output: {args.model_output}"
        f"n_estimators: {args.n_estimators}"
        f"learning_rate: {args.learning_rate}"
        f"base_estimator: {args.base_estimator}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()





