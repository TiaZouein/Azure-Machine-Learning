import argparse

from pathlib import Path

import numpy as np
import pandas as pd
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
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
   

    parser.add_argument('--criterion', type=str, default = "gini",
                        help='critère')


    parser.add_argument('--max_depth', type=int, default=5,
                        help=' Maximum number of levels in tree')


    parser.add_argument('--min_samples_leaf', type=int, default = 1,
                        help='Minimum number of samples required at each leaf node')

    parser.add_argument('--min_samples_split', type=int, default=2,
                        help='Minimum number of samples required to split a node')

    parser.add_argument('--min_impurity_decrease', type=float, default = 0,
                        help='Minimum of impurity decrease')
    
    parser.add_argument('--random_state', type=int, default = 42,
                        help='random_state')

    parser.add_argument('--ccp_alpha', type=float, default = 0,
                        help='ccp_alpha')
    args = parser.parse_args()

    return args


def main(args):
    '''Read train dataset, train model, save trained model'''

    mlflow.sklearn.autolog()
    mlflow.log_param("decision_tree_model", "DecisionTreeClassifier")
    mlflow.log_param('criterion', str(args.criterion))
    mlflow.log_param('max_depth', int(args.max_depth))
    mlflow.log_param('min_samples_leaf', int(args.min_samples_leaf))
    mlflow.log_param('min_samples_split', int(args.min_samples_split))
    mlflow.log_param('min_impurity_decrease', float(args.min_impurity_decrease))
    mlflow.log_param('random_state', int(args.random_state))
    mlflow.log_param('ccp_alpha', float(args.ccp_alpha))
    
    print("reading data")
    df = pd.read_csv(Path(args.input_data) / "data.csv", sep=";")

    Numeric_Var = ["List of numerical variables in your data frame"]
    Cat_Var = ["List of categorical variables in your data frame"]

    print("encoding variables")
    
    label_encoder = LabelEncoder()
    for column in Cat_Var:
        df[column] = label_encoder.fit_transform(df[column])


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

    #Oversampling pour résoudre le problème de déséquilibre des classes
    sampling_strategy = {0: 11710063, 1:4687331}
    ros = RandomOverSampler(random_state=0, sampling_strategy=sampling_strategy)
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train)

    decision_tree_model = DecisionTreeClassifier(criterion = args.criterion,
                                                max_depth = args.max_depth,
                                                min_samples_leaf = args.min_samples_leaf,
                                                min_samples_split = args.min_samples_split,
                                                ccp_alpha = args.ccp_alpha,  
                                                min_impurity_decrease = args.min_impurity_decrease,  
                                                random_state = args.random_state)
    print("training model")                                            
    decision_tree_model = decision_tree_model.fit(X_train_resampled, Y_train_resampled)
    print("predicting values")
    y_decisiontree = decision_tree_model.predict(X_test)
    print("evaluating model")

    accuracyscore = metrics.accuracy_score(Y_test,y_decisiontree)*100
    f1score = metrics.f1_score(Y_test,y_decisiontree)*100
    recallscore = metrics.recall_score(Y_test,y_decisiontree)*100
    specificityscore = metrics.recall_score(Y_test, y_decisiontree, pos_label=0)*100
    precisionscore = metrics.precision_score(Y_test,y_decisiontree)*100


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
    cm = confusion_matrix(Y_test, y_decisiontree)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    cm_display.figure_.savefig('confusion_matrix.png')
    mlflow.log_artifact("confusion_matrix.png")

    ######## COURBE DE ROC
    prob = decision_tree_model.predict_proba(X_test)
    pred = prob[:,1]
    fpr_d, tpr_d, threshold = metrics.roc_curve(Y_test, pred)
    roc_auc = metrics.auc(fpr_d, tpr_d)
    print("AUC Decision Tree=", roc_auc)
    auc_d = roc_auc_score(Y_test, pred)
    plt.title('Courbe de ROC - lightgbm')
    plt.plot(fpr_d, tpr_d, 'b', label = 'AUC = %0.2f' % auc_d)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate - Sensitivity')
    plt.xlabel('False Positive Rate - (1-Specificity)')
    plt.savefig('roc_curve.png')
    mlflow.log_artifact("roc_curve.png")

    feature_importance = decision_tree_model.feature_importances_
    print("Importance des caractéristiques :")
    importance_df = pd.DataFrame({'Caractéristiques': X_train_resampled.columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)

    registered_model_name="sklearn_DT_model"

    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=decision_tree_model,
        registered_model_name=registered_model_name,
        artifact_path=registered_model_name
    )

    # Saving the model to a file
    print("Saving the model via MLFlow")
    mlflow.sklearn.save_model(
        sk_model=decision_tree_model,
        path=os.path.join(registered_model_name, "DT_model"),
    )



if __name__ == "__main__":
    
    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"input_data: {args.input_data}",
        f"model_output: {args.model_output}"
        f"criterion: {args.criterion}"   
        f"max_depth: {args.max_depth}"
        f"min_samples_leaf: {args.min_samples_leaf}"
        f"min_samples_split: {args.min_samples_split}"
        f"random_state: {args.random_state}"
        f"ccp_alpha: {args.ccp_alpha}"
        f"min_impurity_decrease: {args.min_impurity_decrease}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()





