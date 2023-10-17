import argparse

from pathlib import Path

import numpy as np
import pandas as pd
import os
import xgboost as xgb
from xgboost import XGBClassifier 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier 
from keras.callbacks import EarlyStopping

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from keras.layers import Input, Embedding, Flatten, Dense, BatchNormalization, ReLU, Dropout, Concatenate
from keras.models import Model
from keras.models import Sequential  
from keras.optimizers import Adam, Adadelta

import matplotlib.pyplot as plt
from collections import Counter
import lightgbm as lgbm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
import mlflow
import mlflow.sklearn


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to data")
    parser.add_argument("--model_output", type=str, help="Path to model")
    
    # for lighgbm
    parser.add_argument('--num_leaves', type=int, default = 100,
                        help='Maximum number of leaves in one tree, should not be high, defaul is 31')

    parser.add_argument('--objective', type=str, default = 'binary',
                        help='loss function')

    parser.add_argument('--num_boost_round', type=int, default=100,
                        help='number of boosting iterations')

    parser.add_argument('--learning_rate_lgbm', type=float, default=0.1,
                        help='learning_rate used to do the new prediction')
    
    
    parser.add_argument('--max_depth_lgbm', type=int, default = 100,
                        help='max_depth')

    parser.add_argument('--min_data_in_leaf', type=int, default=50,
                        help='Minimum number of samples required in a terminal leaf')
    
    parser.add_argument('--min_gain_to_split', type=float, default = 0.0001,
                        help='random_state')

    parser.add_argument('--max_bin', type=int, default = 255,
                        help='small number of bins may reduce training accuracy but may increase general power ')
        
    parser.add_argument('--categorical_feature', type=str, default =['origine', 'destination','voiture', 'duplex', 'place', 'classe_voiture', 'lib_article',
    'lib_canal_dist','type_point_vente', 'lib_canal_av', 'lib_motif_av','lib_nature_av',
    'cd_poste_av', 'lib_transporteur', 'list_type_paiement','date_prestation_bin', 'date_naissance_bin'],
                        help='specify categorical features')

    parser.add_argument('--num_class', type=int, default = 2,
                        help='random_state')
    
    #parser.add_argument('--early_stopping_round', type=int, default = 50,
                       # help='early_stopping') 


    parser.add_argument('--data_sample_strategy', type=str, default = 'goss',
                        help='lightgbm samples data after each weak learner, either bagging (random sampling with replacement, either goss, wich is one of the most features of lightgbm that makes it powerful, by default implementation of lightgbm, it is bagging, that is why i have to specify that i want goss method !')  

    parser.add_argument('--lambda_l2', type=float, default = 0.1,
                        help='l2 regularisation')   
    
    parser.add_argument('--top_rate', type=float, default = 0.8,
                        help='the retain ratio of large gradient data')

    parser.add_argument('--other_rate', type=float, default = 0.2,
                        help='the retain ratio of small gradient data') 

    # for Gradient Boosting  
    parser.add_argument('--n_estimators_gb', type=int, default = 1,
                        help='Number of trees')
    parser.add_argument('--learning_rate_gb', type=float, default = 0.01,
                        help='learning_rate')
    parser.add_argument('--max_depth_gb', type=int, default=10,
                        help=' Maximum number of levels in tree')
    parser.add_argument('--max_features', type=float, default=1.0,
                        help='Number of features to consider at every split')    
    parser.add_argument('--subsample_gb', type=float, default=0.7,
                        help='Number of observations taken by bootstrap to every estimator')
    parser.add_argument('--min_samples_leaf', type=int, default = 300,
                        help='Minimum number of samples required at each leaf node')
    parser.add_argument('--min_samples_split', type=int, default=300,
                        help='Minimum number of samples required to split a node')
    parser.add_argument('--min_impurity_decrease', type=float, default = 0.01,
                        help='Minimum of impurity decrease')    
    parser.add_argument('--random_state', type=int, default = 42,
                        help='random_state')
    parser.add_argument('--ccp_alpha', type=float, default = 0.5,
                        help='ccp_alpha')
    
    # for xgboost
    parser.add_argument("--n_estimators_xgb", type=int,  default = 10, help="nombre de weak learners")   
    parser.add_argument('--eta', type=float, default = 0.3,
                        help='learning_rate')
    parser.add_argument('--gamma', type=int, default = 0,
                        help='valeur à enlever du Gain')
    parser.add_argument('--max_depth_xgb', type=int, default=6,
                        help=' Maximum number of levels in tree')
    parser.add_argument('--min_child_weight', type=int, default=1, 
                        help='Cover to be satisfied by the residuals in the terminal leaf') ## 1 for classification doen't mean 1 residual in the leaf !    
    parser.add_argument('--subsample_xgb', type=float, default=1,
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



    # for adaboost :
    #parser.add_argument("--base_estimator", type=object, default = DecisionTreeClassifier())
   
    #parser.add_argument('--n_estimators', type=int, default = 10000,
                        #help='Number of weak learners')
    
    # for neural network
    parser.add_argument('--learning_rate_MLP', type=float, default=0.10, help='percentage of dropout')
    parser.add_argument('--activation', type=str, default='relu', help='activation function')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--loss', type=str, default='binary_crossentropy', help='loss')
    parser.add_argument('--rate', type=float, default=0.1, help='Dropout layer')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--epochs', type=int, default=5, help='epochs')
    args = parser.parse_args()

    return args


def main(args):
    '''Read train dataset, train model, save trained model'''

    mlflow.sklearn.autolog()
    mlflow.log_param("lightgbm", "Lightgbm")
    mlflow.log_param('num_leaves', int(args.num_leaves))
    mlflow.log_param('objective', str(args.objective))
    mlflow.log_param('num_boost_round', str(args.num_boost_round))
    mlflow.log_param('learning_rate_lgbm', float(args.learning_rate_lgbm))
    #mlflow.log_param('num_threads', int(args.num_threads))
    mlflow.log_param('max_depth_lgbm', int(args.max_depth_lgbm))
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

    # adaboost
    #mlflow.log_param("adaboost_model", "AdaBoostClassifier")    
    #mlflow.log_param('n_estimators', int(args.n_estimators))

    # NN
    mlflow.log_param("MLP_model", "MLP")
    mlflow.log_param('learning_rate_MLP', float(args.learning_rate_MLP))
    mlflow.log_param('rate', float(args.rate))
    mlflow.log_param('activation', str(args.activation))
    mlflow.log_param('optimizer', str(args.activation))
    mlflow.log_param('loss', str(args.loss))
    mlflow.log_param('epochs', int(args.epochs))
    
    # GRADIENT BOOSITNG
    mlflow.log_param('n_estimators_gb', int(args.n_estimators_gb))
    mlflow.log_param('max_depth_gb', int(args.max_depth_gb))
    mlflow.log_param('max_features', float(args.max_features))
    mlflow.log_param('subsample_gb', float(args.subsample_gb))
    mlflow.log_param('min_samples_leaf', int(args.min_samples_leaf))
    mlflow.log_param('min_samples_split', int(args.min_samples_split))
    mlflow.log_param('min_impurity_decrease', float(args.min_impurity_decrease))
    mlflow.log_param('random_state', int(args.random_state))
    mlflow.log_param('ccp_alpha', float(args.ccp_alpha))
    mlflow.log_param('learning_rate_gb', float(args.learning_rate_gb))

    # XGBOOST
    mlflow.log_param('eta', float(args.eta))
    mlflow.log_param('n_estimators_xgb', float(args.n_estimators_xgb))
    mlflow.log_param('gamma', int(args.gamma))
    mlflow.log_param('max_depth_xgb', int(args.max_depth_xgb))
    mlflow.log_param('min_child_weight', int(args.min_child_weight))
    mlflow.log_param('subsample_xgb', float(args.subsample_xgb))
    mlflow.log_param('reg_lambda', int(args.reg_lambda))
    mlflow.log_param('alpha', int(args.alpha))
    mlflow.log_param('num_parallel_tree', int(args.num_parallel_tree))
    mlflow.log_param('importance_type', str(args.importance_type))
    mlflow.log_param('tree_method', str(args.tree_method)) 
    #mlflow.log_param('nthread', float(args.nthread)) 
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
    #scaler = StandardScaler()
    scaler = MaxAbsScaler()
    df[Numeric_Var] = scaler.fit_transform(df[Numeric_Var])

    print("splitting")
    X = df.drop("fraud", axis=1)
    Y = df["fraud"]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=15,shuffle=True,stratify=Y)  

    X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.1,random_state=15,shuffle=True,stratify=Y_train)  
    print("Y_train:\n",  Y_train.value_counts(), 
    "\nY_test:\n", Y_test.value_counts() 
    ,"\nY_valid:\n", Y_val.value_counts())



    print("SAMPLING")
    sampling_strategy = {0: 12044636, 1:4687331}  
    ros = RandomOverSampler(random_state=0, sampling_strategy=sampling_strategy)
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train )
    
    print("Class distribution after under-sampling:")
    print("X_train_resampled shape:", X_train_resampled.shape)
    print("Y_train_resampled shape:", Y_train_resampled.shape)
    print("Y_train_Count:", Y_train_resampled.value_counts())
    print(sorted(Counter(Y_train_resampled).items()))

    # defining models
    categorical_indices = [X_train.columns.get_loc(cat_feature) for cat_feature in Cat_Var]
    lgbm_model = lgbm.LGBMClassifier(num_leaves= args.num_leaves,
                                    objective = args.objective,                                            
                                    max_depth = args.max_depth_lgbm,
                                    num_boost_round  = args.num_boost_round ,
                                    learning_rate = args.learning_rate_lgbm,
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

    gradient_boosting_model = GradientBoostingClassifier(n_estimators= args.n_estimators_gb,
                                    learning_rate = args.learning_rate_gb,                                            
                                    max_depth = args.max_depth_gb,
                                    max_features = args.max_features,
                                    subsample = args.subsample_gb,
                                    min_samples_leaf = args.min_samples_leaf,
                                    min_samples_split = args.min_samples_split,
                                    ccp_alpha = args.ccp_alpha,  
                                    min_impurity_decrease = args.min_impurity_decrease,  
                                    random_state = args.random_state)
    

    xgboost_model = xgb.XGBClassifier(eta= args.eta,
                                    gamma = args.gamma,                                            
                                    max_depth = args.max_depth_xgb,
                                    min_child_weight = args.min_child_weight,
                                    subsample = args.subsample_xgb,
                                    reg_lambda = args.reg_lambda,
                                    num_parallel_tree = args.num_parallel_tree,
                                    alpha = args.alpha,  
                                    tree_method = args.tree_method,
                                    n_estimators = args.n_estimators_xgb,
                                    importance_type = args.importance_type,
                                    #nthread = args.nthread,
                                    enable_categorical = args.enable_categorical)
    
    #adaboost_model = AdaBoostClassifier(n_estimators= args.n_estimators,
                                                #base_estimator = base_classifier,
                                                #learning_rate = args.learning_rate,)
                                            
    

    print("training models on the sampled data")                                          
    lgbm_model = lgbm_model.fit(X_train_resampled, Y_train_resampled)  
    xgboost_model = xgboost_model.fit(X_train_resampled, Y_train_resampled)
    gradient_boosting_model = gradient_boosting_model.fit(X_train_resampled, Y_train_resampled)
    
   
    print("predicting values on validation data")
    y_lgbm = lgbm_model.predict(X_val)
    y_xgboost = xgboost_model.predict(X_val)
    y_gradientboosting  = gradient_boosting_model.predict(X_val)


    print("training the meta model")
    X_val_meta = np.column_stack((y_lgbm, y_xgboost, y_gradientboosting))

    model_MLP = Sequential()
    model_MLP.add(Dense(256,input_shape=(3,))) # input shape doit etre egal au nombre de base models
    model_MLP.add(BatchNormalization())
    model_MLP.add(Dropout(rate=args.rate))
    model_MLP.add(ReLU())
    model_MLP.add(Dense(128)) 
    model_MLP.add(BatchNormalization())
    model_MLP.add(Dropout(rate=args.rate))
    model_MLP.add(ReLU())
    model_MLP.add(Dense(64)) 
    model_MLP.add(BatchNormalization())
    model_MLP.add(Dropout(rate=args.rate))
    model_MLP.add(ReLU())
    model_MLP.add(Dense(32)) 
    model_MLP.add(BatchNormalization())
    model_MLP.add(Dropout(rate=args.rate))
    model_MLP.add(ReLU())
    model_MLP.add(Dense(1, activation='sigmoid'))
    model_MLP.summary()
    #opt = Adam(lr=args.lr)
    model_MLP.compile(optimizer=Adadelta(learning_rate=args.learning_rate_MLP), loss=args.loss, metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 3)

    hist_MLP = model_MLP.fit(X_val_meta, Y_val, batch_size=args.batch_size,
              epochs=args.epochs,
              shuffle=True,
              validation_split=0.1,
              callbacks = [es])

    print("predicting on new data wich is X_test")
    y_lgbm_test = lgbm_model.predict(X_test)
    y_gradientboosting_test = gradient_boosting_model.predict(X_test)
    y_xgboost_test = xgboost_model.predict(X_test)

    print("Combining the predictions of the base models into a single feature matrix")
    X_new_meta = np.column_stack((y_lgbm_test, y_gradientboosting_test,y_xgboost_test))

    print("prediction using the meta-model on X_meta_test")
    y_test_pred = model_MLP.predict(X_new_meta)

    y_test_pred_labels = (y_test_pred > 0.5).astype(int)

    # Visualize results
    cm = confusion_matrix(Y_test, y_test_pred_labels)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    cm_display.figure_.savefig('confusion_matrix-MLP.png')
    mlflow.log_artifact("confusion_matrix-MLP.png")

    accuracyscore = metrics.accuracy_score(Y_test,y_test_pred_labels)*100
    f1score = metrics.f1_score(Y_test,y_test_pred_labels)*100
    recallscore = metrics.recall_score(Y_test,y_test_pred_labels)*100
    specificityscore = metrics.recall_score(Y_test, y_test_pred_labels, pos_label=0)*100
    precisionscore = metrics.precision_score(Y_test,y_test_pred_labels)*100


    mlflow.log_metric("accuracy_score", accuracyscore)
    mlflow.log_metric("f1_score", f1score)
    mlflow.log_metric("precision_score", precisionscore)
    mlflow.log_metric("recall_score", recallscore)
    mlflow.log_metric("specificity_score", specificityscore)
    print ("Accuracy=", accuracyscore)
    print ("f1.score=", f1score)
    print ("precision.score=", precisionscore)
    print ("recall_score=", recallscore)
    print("Specificity=", specificityscore)

    
    ######## COURBE DE ROC
    fpr_d, tpr_d, threshold = metrics.roc_curve(Y_test, y_test_pred)
    roc_auc = metrics.auc(fpr_d, tpr_d)
    print("AUC MLP=", roc_auc)
    auc_d = roc_auc_score(Y_test, y_test_pred)
    plt.title('Courbe de ROC - MLP')
    plt.plot(fpr_d, tpr_d, 'b', label = 'AUC = %0.2f' % auc_d)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate - Sensitivity')
    plt.xlabel('False Positive Rate - (1-Specificity)')
    plt.savefig('roc_curve.png')
    mlflow.log_artifact("roc_curve.png")

    hist_MLP.history.keys()
    plt.figure(2)
    plt.plot(hist_MLP.history['loss'], label='training set',marker='o', linestyle='solid',linewidth=1, markersize=6)
    plt.plot(hist_MLP.history['val_loss'], label='validation set',marker='o', linestyle='solid',linewidth=1, markersize=6)
    plt.title("MLP-model loss")
    plt.xlabel('Nombre d\'epochs')
    plt.ylabel('Total Loss')
    plt.legend(bbox_to_anchor=( 1.35, 1.))
    plt.savefig('Loss.png')
    mlflow.log_artifact("Loss.png")



    # Visualize results
    cm = confusion_matrix(Y_test, y_lgbm_test)
    plt.figure(3)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    cm_display.figure_.savefig('confusion_matrix-lgbm.png')
    mlflow.log_artifact("confusion_matrix-lgbm.png")

    cm = confusion_matrix(Y_test, y_gradientboosting_test)
    plt.figure(4)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    cm_display.figure_.savefig('confusion_matrix-gradientboosting.png')
    mlflow.log_artifact("confusion_matrix-gradientboosting.png")

    cm = confusion_matrix(Y_test, y_xgboost_test)
    plt.figure(5)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    cm_display.figure_.savefig('confusion_matrix-xgboost.png')
    mlflow.log_artifact("confusion_matrix-xgboost.png")

    plt.figure(figsize=(8, 6))
    plt.figure(6)
    plt.plot(hist_MLP.history['loss'], label='Train')
    plt.plot(hist_MLP.history['val_loss'], label='Validation')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.title('Loss_Curve')
    plt.savefig('loss_curve.png')
    mlflow.log_artifact("loss_curve.png")


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
        f"max_depth_lgbm: {args.max_depth_lgbm}"
        f"num_boost_round : {args.num_boost_round }"
        f"learning_rate_lgbm: {args.learning_rate_lgbm}"
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
        #f"base_estimator: {args.base_estimator}"   

        # MLP 
        f"learning_rate_MLP: {args.learning_rate_MLP}"
        f"rate: {args.rate}"
        f"activation: {args.activation}"
        f"optimizer: {args.optimizer}"
        f"loss: {args.loss}"  

        # GRADIENT BOOSTING
        f"n_estimators_gb: {args.n_estimators_gb}"
        f"learning_rate_gb: {args.learning_rate_gb}"
        f"max_features: {args.max_features}"
        f"max_depth_gb: {args.max_depth_gb}"
        f"subsample_gb: {args.subsample_gb}"
        f"min_samples_leaf: {args.min_samples_leaf}"
        f"min_samples_split: {args.min_samples_split}"
        f"random_state: {args.random_state}"
        f"ccp_alpha: {args.ccp_alpha}"
        f"min_impurity_decrease: {args.min_impurity_decrease}"

        # XGBOOST
        f"enable_categorical: {args.enable_categorical}"
        f"n_estimators_xgb: {args.n_estimators_xgb}"
        f"eta: {args.eta}"
        f"gamma: {args.gamma}"
        f"max_depth_xgb: {args.max_depth_xgb}"
        f"min_child_weight: {args.min_child_weight}"
        f"subsample_xgb: {args.subsample_xgb}"
        f"max_depth_xgb: {args.max_depth_xgb}"
        f"subsample_xgb: {args.subsample_xgb}"
        f"reg_lambda: {args.reg_lambda}"
        f"num_parallel_tree: {args.num_parallel_tree}"
        f"alpha: {args.alpha}"
        f"tree_method: {args.tree_method}"
        #f"nthread: {args.nthread}"
        f"importance_type: {args.importance_type}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()





