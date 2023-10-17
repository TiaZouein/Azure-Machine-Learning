import argparse

from pathlib import Path
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from collections import Counter

import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_selection import SequentialFeatureSelector
from collections import Counter
#import lightgbm as lgbm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
import mlflow
import mlflow.sklearn


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to data")
    parser.add_argument("--model_output", type=str, help="Path to model")
   
    parser.add_argument('--num_leaves', type=int, default = 100,
                        help='Maximum number of leaves in one tree, should not be high, defaul is 31')

    parser.add_argument('--objective', type=str, default = 'binary',
                        help='loss function')

    parser.add_argument('--num_iterations', type=int, default=100,
                        help='number of boosting iterations')

    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning_rate used to do the new prediction')
    
    #parser.add_argument('--num_threads', type=int, default=1,
                      #  help='Same as n_jobs')

    parser.add_argument('--max_depth', type=int, default = 100,
                        help='max_depth')

    parser.add_argument('--min_data_in_leaf', type=int, default=50,
                        help='Minimum number of samples required in a terminal leaf')
    
    parser.add_argument('--min_gain_to_split', type=float, default = 0.0001,
                        help='random_state')

    parser.add_argument('--max_bin', type=int, default = 255,
                        help='small number of bins may reduce training accuracy but may increase general power ')
        
    parser.add_argument('--categorical_feature', type=str, default =['put your default'],
                        help='specify categorical features')

    parser.add_argument('--num_class', type=int, default = 2,
                        help='random_state')
    
    #parser.add_argument('--early_stopping_round', type=int, default = 50,
                       # help='early_stopping') 


    parser.add_argument('--data_sample_strategy', type=str, default = 'goss',
                        help='lightgbm samples data after each weak learner, either bagging (random sampling with replacement, either goss, wich is one of the most important features of lightgbm that makes it powerful, by default implementation of lightgbm, it is bagging, that is why i have to specify that i want goss method !')  

    parser.add_argument('--lambda_l2', type=float, default = 0.1,
                        help='l2 regularisation')   
    
    parser.add_argument('--top_rate', type=float, default = 0.8,
                        help='the retain ratio of large gradient data')

    parser.add_argument('--other_rate', type=float, default = 0.2,
                        help='the retain ratio of small gradient data') 
    
    

    args = parser.parse_args()

    return args


def main(args):
    '''Read train dataset, train model, save trained model'''

    mlflow.sklearn.autolog()
    mlflow.log_param("lightgbm", "Lightgbm")
    mlflow.log_param('num_leaves', int(args.num_leaves))
    mlflow.log_param('objective', str(args.objective))
    mlflow.log_param('num_iterations', str(args.num_iterations))
    mlflow.log_param('learning_rate', float(args.learning_rate))
    #mlflow.log_param('num_threads', int(args.num_threads))
    mlflow.log_param('max_depth', int(args.max_depth))
    mlflow.log_param('min_data_in_leaf', int(args.min_data_in_leaf))
    mlflow.log_param('min_gain_to_split', float(args.min_gain_to_split))
    mlflow.log_param('max_bin', int(args.max_bin))
    mlflow.log_param('categorical_feature', str(args.categorical_feature))
    mlflow.log_param('num_class', int(args.num_class))
    #mlflow.log_param('early_stopping_round', int(args.early_stopping_round))
    mlflow.log_param('data_sample_strategy', str(args.data_sample_strategy))
    mlflow.log_param('lambda_l2', float(args.lambda_l2))
    mlflow.log_param('top_rate', float(args.top_rate))
    mlflow.log_param('other_rate', float(args.other_rate))
    
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
    
    #scaler = StandardScaler()
    scaler = MaxAbsScaler()
    df[Numeric_Var] = scaler.fit_transform(df[Numeric_Var])
    
    print("splitting")
    X = df.drop("fraud", axis=1)
    Y = df["fraud"]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=15,shuffle=True,stratify=Y)  

    print("Y_train:\n",  Y_train.value_counts(), 
      "\nY_test:\n", Y_test.value_counts())



    print("SAMPLING")
    categorical_indices = [df.columns.get_loc(cat_feature) for cat_feature in Cat_Var]
    lgbm_model = LGBMClassifier(num_leaves= args.num_leaves,
                                    objective = args.objective,                                            
                                    max_depth = args.max_depth,
                                    num_iterations = args.num_iterations,
                                    learning_rate = args.learning_rate,
                                    #num_threads = args.num_threads,
                                    min_data_in_leaf = args.min_data_in_leaf,
                                    max_bin = args.max_bin,  
                                    categorical_feature = categorical_indices,
                                    #early_stopping_round = args.early_stopping_round,
                                    data_sample_strategy = args.data_sample_strategy,
                                    lambda_l2 = args.lambda_l2,
                                    top_rate = args.top_rate,
                                    other_rate = args.other_rate,
                                    )

    sampling_strategy = {0: 11710063, 1:4687331}
    ros = RandomOverSampler(random_state=0, sampling_strategy=sampling_strategy)
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train )
    
    print("Class distribution after under-sampling:")
    print("X_train_resampled shape:", X_train_resampled.shape)
    print("Y_train_resampled shape:", Y_train_resampled.shape)
    print("Y_train_Count:", Y_train_resampled.value_counts())
    print(sorted(Counter(Y_train_resampled).items()))

    print("training model on the sampled data")                                            
    lgbm_model = lgbm_model.fit(X_train_resampled, Y_train_resampled)
    print("predicting values")
    y_lgbm = lgbm_model.predict(X_test)
    print("evaluating model on the sampled data")
    print(classification_report(Y_test, y_lgbm))
    
    # Calculate and log accuracy on train and test data
    train_accuracy = lgbm_model.score(X_train_resampled, Y_train_resampled)*100
    test_accuracy = metrics.accuracy_score(Y_test, y_lgbm)*100
    
    f1score = metrics.f1_score(Y_test,y_lgbm)*100
    recallscore = metrics.recall_score(Y_test,y_lgbm)*100
    specificityscore = metrics.recall_score(Y_test, y_lgbm, pos_label=0)*100
    precisionscore = metrics.precision_score(Y_test,y_lgbm)*100
    
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("f1_score", f1score)
    mlflow.log_metric("precision_score", precisionscore)
    mlflow.log_metric("recall_score", recallscore)
    mlflow.log_metric("specificity_score", specificityscore)
    print ("train_accuracy=", train_accuracy)
    print ("test_accuracy=", test_accuracy)
    print ("f1.score_Sampled==", f1score)
    print ("Precision_Sampled==", precisionscore)
    print ("Recall_Sampled==", recallscore)
    print("Specificity_Sampled==", specificityscore)

    # Visualize results
    cm = confusion_matrix(Y_test, y_lgbm)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    cm_display.figure_.savefig('confusion_matrix.png')
    mlflow.log_artifact("confusion_matrix.png")

    ######## COURBE DE ROC
    # calculate the fpr and tpr for all thresholds of the classification
    prob = lgbm_model.predict_proba(X_test)
    pred = prob[:,1]
    fpr_d, tpr_d, threshold = metrics.roc_curve(Y_test, pred)
    roc_auc = metrics.auc(fpr_d, tpr_d)
    print("AUC lightgbm=", roc_auc)
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


    feature_importance = lgbm_model.feature_importances_
    print("Importance des caractéristiques :")
    importance_df = pd.DataFrame({'Caractéristiques': X_train_resampled.columns, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)

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
        f"input_data: {args.input_data}",
        f"model_output: {args.model_output}"
        f"num_leaves: {args.num_leaves}"
        f"objective: {args.objective}"
        f"max_depth: {args.max_depth}"
        f"num_iterations: {args.num_iterations}"
        f"learning_rate: {args.learning_rate}"
        #f"num_threads: {args.num_threads}"
        f"min_data_in_leaf: {args.min_data_in_leaf}"
        f"max_bin: {args.max_bin}"
        f"categorical_feature: {args.categorical_feature}"
        f"num_class: {args.num_class}"          
        #f"early_stopping_round: {args.early_stopping_round}"          
        f"data_sample_strategy: {args.data_sample_strategy}"          
        f"lambda_l2: {args.lambda_l2}"          
        f"top_rate: {args.top_rate}"          
        f"other_rate: {args.other_rate}"          
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()





