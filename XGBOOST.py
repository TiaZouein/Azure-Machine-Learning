import argparse

from pathlib import Path
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.over_sampling import SMOTE, ADASYN

from collections import Counter

import xgboost as xgb
from xgboost import XGBClassifier 

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to data")
    parser.add_argument("--model_output", type=str, help="Path to model")

    parser.add_argument("--n_estimators", type=int,  default = 10, help="nombre de weak learners")
   
    parser.add_argument('--eta', type=float, default = 0.3,
                        help='learning_rate')

    parser.add_argument('--gamma', type=int, default = 0,
                        help='valeur à enlever du Gain')

    parser.add_argument('--max_depth', type=int, default=6,
                        help=' Maximum number of levels in tree')

    parser.add_argument('--min_child_weight', type=int, default=1, 
                        help='Cover to be satisfied by the residuals in the terminal leaf') ## 1 for classification doen't mean 1 residual in the leaf !
    
    parser.add_argument('--subsample', type=float, default=1,
                        help='Number of observations taken by each weak learner')

    parser.add_argument('--reg_lambda', type=int, default = 1,
                        help='l2 Regularisation hyperparameter')

    parser.add_argument('--alpha', type=int, default=0,
                        help='l1 Regularisation hyperparameter')

    parser.add_argument('--num_parallel_tree', type=int, default = 1,
                        help='Nombres des arbres pour chaque weak learner, when >1, each weak leaner is like a random forest, when 1, it is like a decision tree')

    
    parser.add_argument('--tree_method', type=str, default = 'hist',
                        help='random_state')
    
    parser.add_argument('--importance_type', type=str, default = 'gain',
                        help='feature_importance')

    parser.add_argument('--enable_categorical', type=bool, default = True,
                        help='handling categorical variables')

    parser.add_argument('--nthread', type=int, default = 1,
                        help='same as n_jobs')
                         

    args = parser.parse_args()

    return args


def main(args):
    '''Read train dataset, train model, save trained model'''

    mlflow.sklearn.autolog()
    mlflow.log_param("xbgoost_model", "xgbClassifier")
    mlflow.log_param('eta', float(args.eta))
    mlflow.log_param('n_estimators', float(args.n_estimators))
    mlflow.log_param('gamma', int(args.gamma))
    mlflow.log_param('max_depth', int(args.max_depth))
    mlflow.log_param('min_child_weight', int(args.min_child_weight))
    mlflow.log_param('subsample', float(args.subsample))
    mlflow.log_param('reg_lambda', int(args.reg_lambda))
    mlflow.log_param('alpha', int(args.alpha))
    mlflow.log_param('num_parallel_tree', int(args.num_parallel_tree))
    mlflow.log_param('importance_type', str(args.importance_type))
    mlflow.log_param('tree_method', str(args.tree_method)) 
    mlflow.log_param('nthread', float(args.nthread)) 
    mlflow.log_param('enable_categorical', bool(args.enable_categorical)) 
    
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



    print("SAMPLING")
    xgboost_model = xgb.XGBClassifier(eta= args.eta,
                                    gamma = args.gamma,                                            
                                    max_depth = args.max_depth,
                                    min_child_weight = args.min_child_weight,
                                    subsample = args.subsample,
                                    reg_lambda = args.reg_lambda,
                                    num_parallel_tree = args.num_parallel_tree,
                                    alpha = args.alpha,  
                                    tree_method = args.tree_method,
                                    n_estimators = args.n_estimators,
                                    importance_type = args.importance_type,
                                    nthread = args.nthread,
                                    enable_categorical = args.enable_categorical)
    
    sampling_strategy = {0: 11710063, 1:4687331}
    ros = RandomOverSampler(random_state=0, sampling_strategy=sampling_strategy)
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train)
    
    print("Class distribution after over-sampling:")
    print("X_train_resampled shape:", X_train_resampled.shape)
    print("Y_train_resampled shape:", Y_train_resampled.shape)
    print("Y_train_Count:", Y_train_resampled.value_counts())
    print(sorted(Counter(Y_train_resampled).items()))

    print("training model on the sampled data")                                            
    xgboost_model = xgboost_model.fit(X_train_resampled, Y_train_resampled)
    print("predicting values")
    y_xgboost = xgboost_model.predict(X_test)
    
    accuracyscore = metrics.accuracy_score(Y_test,y_xgboost)*100
    f1score = metrics.f1_score(Y_test,y_xgboost)*100
    recallscore = metrics.recall_score(Y_test,y_xgboost)*100
    specificityscore = metrics.recall_score(Y_test, y_xgboost, pos_label=0)*100
    precisionscore = metrics.precision_score(Y_test,y_xgboost)*100

    mlflow.log_metric("accuracy_score", accuracyscore)
    mlflow.log_metric("f1_score", f1score)
    mlflow.log_metric("precision_score", precisionscore)
    mlflow.log_metric("recall_score", recallscore)
    mlflow.log_metric("specificity_score", specificityscore)
    print ("Accuracy_Sampled=", accuracyscore)
    print ("f1.score_Sampled==", f1score)
    print ("Precision_Sampled==", precisionscore)
    print ("Recall_Sampled==", recallscore)
    print("Specificity_Sampled==", specificityscore)

    #GAIN : C'est la contribution moyenne de chaque fonctionnalité à la réduction de l'erreur lorsqu'elle est utilisée dans les arbres de décision.Comprendre comment chaque fonctionnalité contribue à améliorer la qualité des prédictions.
    gain_importance = xgboost_model.get_booster().get_score(importance_type='gain')

     # le nombre de fois que la fonctionnalité est sélectionnée pour effectuer des décisions dans l'ensemble des arbres
    weight_importance = xgboost_model.get_booster().get_score(importance_type='weight')

    sorted_gain_importance = sorted(gain_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_weight_importance = sorted(weight_importance.items(), key=lambda x: x[1], reverse=True)

    print("Feature Importance based on Gain:")
    for feature, score in sorted_gain_importance:
        print(f"{feature}: {score}")

    print("\nFeature Importance based on Weight:")
    for feature, score in sorted_weight_importance:
        print(f"{feature}: {score}")

    # Visualize results
    cm = confusion_matrix(Y_test, y_xgboost)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    cm_display.figure_.savefig('confusion_matrix.png')
    mlflow.log_artifact("confusion_matrix.png")

    ######## COURBE DE ROC
    # calculate the fpr and tpr for all thresholds of the classification
    prob = xgboost_model.predict_proba(X_test)
    pred = prob[:,1]
    fpr_d, tpr_d, threshold = metrics.roc_curve(Y_test, pred)
    roc_auc = metrics.auc(fpr_d, tpr_d)
    print("AUC Xgboost=", roc_auc)
    auc_d = roc_auc_score(Y_test, pred)
    plt.title('Courbe de ROC - Adaboost')
    plt.figure(1)
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

    precision, recall, _ = precision_recall_curve(Y_test, xgboost_model.predict_proba(X_test)[:, 1])
    plt.figure(2)
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
    

    feature_importances = xgboost_model.feature_importances_

    feature_names = X_train_resampled.columns

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(3)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Importance des Variables')
    plt.xlabel('Importance')
    plt.ylabel('Variable')
    plt.savefig('Importance des Variables.png')
    mlflow.log_artifact("Importance des Variables.png")


    registered_model_name="sklearn-Gradient_Boosting_Sampled-model"

    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=gradient_boosting_sampled_model,
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name
    )

    # Saving the model to a file
    print("Saving the model via MLFlow")
    mlflow.sklearn.save_model(
        sk_model=gradient_boosting_sampled_model,
        path=os.path.join(registered_model_name, "GB_FACTORIZE_model"),
    )



if __name__ == "__main__":
    
    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"input_data: {args.input_data}"  
        f"model_output: {args.model_output}"
        f"enable_categorical: {args.enable_categorical}"
        f"n_estimators: {args.n_estimators}"
        f"eta: {args.eta}"
        f"gamma: {args.gamma}"
        f"max_depth: {args.max_depth}"
        f"min_child_weight: {args.min_child_weight}"
        f"subsample: {args.subsample}"
        f"max_depth: {args.max_depth}"
        f"subsample: {args.subsample}"
        f"reg_lambda: {args.reg_lambda}"
        f"num_parallel_tree: {args.num_parallel_tree}"
        f"alpha: {args.alpha}"
        f"tree_method: {args.tree_method}"
        f"nthread: {args.nthread}"
        f"importance_type: {args.importance_type}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()





