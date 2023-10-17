import argparse

from pathlib import Path

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.layers import Input, Embedding, Flatten, Dense, Reshape, BatchNormalization, ReLU, Dropout, Concatenate
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

from sklearn.preprocessing import StandardScaler

from keras.optimizers import Adam

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
    parser.add_argument("--units", type=int, default=256)
    parser.add_argument('--rate', type=float, default=0.10, help='percentage of dropout')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--activation', type=str, default='relu', help='activation function')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--loss', type=str, default='binary_crossentropy', help='loss')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    args = parser.parse_args()
    return args


def main(args):
    '''Read train dataset, train model, save trained model'''

    mlflow.sklearn.autolog()
    mlflow.log_param("MLP_model", "MLP")
    
    mlflow.log_param('units', int(args.units))
    mlflow.log_param('rate', float(args.rate))
    mlflow.log_param('learning_rate', float(args.learning_rate))
    mlflow.log_param('activation', str(args.activation))
    mlflow.log_param('optimizer', str(args.activation))
    mlflow.log_param('loss', str(args.loss))
   
    
    print("reading data")
    df = pd.read_csv(Path(args.input_data) / "data.csv", sep=";")
       
    Numeric_Var = ["List of numerical variables in your data frame"]
    Cat_Var = ["List of categorical variables in your data frame"]

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
    
    print("sampling")
    sampling_strategy = {0: 11710063, 1:4687331}
    ros = RandomOverSampler(random_state=0, sampling_strategy=sampling_strategy)
    X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train)

    print("Class distribution after under-sampling:")
    print("X_train_resampled shape:", X_train_resampled.shape)
    print("Y_train_resampled shape:", Y_train_resampled.shape)
    print("Y_train_Count:", Y_train_resampled.value_counts())
    print(sorted(Counter(Y_train_resampled).items()))

    col_vals_dict = {}
    for c in df.columns:
        if c in Cat_Var:
            try:
                unique_values = list(df[c].unique())
                col_vals_dict[c] = unique_values
            except Exception as e:
                print(f"Error occurred for column '{c}': {str(e)}")

    
    print("embedding entity")
    col_vals_dict = {c: list(df[c].unique()) for c in df.columns if c in Cat_Var}
    embed_cols = []
    len_embed_cols = []
    for c in Cat_Var:
        embed_cols.append(c)
        len_embed_cols.append(len(col_vals_dict[c]))
        print(c + ': %d values' % len(col_vals_dict[c]))
    print('\n Number of features to be embeded  :', len(embed_cols))

    def build_embedding_network(len_embed_cols):    
        
        num_categories_per_variable = []
        categorical_inputs = []
        embedding_layers = []

        for num_categories in len_embed_cols:
            categorical_input = Input(shape=(1,), dtype='int32')
            embedding_dim = min(50, num_categories// 2) 
            print("embedding_dim", embedding_dim)
            embedding_layer = Embedding(input_dim=num_categories, output_dim=embedding_dim, input_length=1)(categorical_input)
            embedding_layer = Flatten()(embedding_layer)  # Flatten the embedding layer
            embedding_layers.append(embedding_layer)
            categorical_inputs.append(categorical_input)
        num_numerical_features = X_train_resampled.shape[1] - len(Cat_Var)  # -1 pour la fraude ! 
        numerical_input = Input(shape=(num_numerical_features,))
        embedding_layers.append(numerical_input)    

        x = Concatenate()(embedding_layers) # all layers (categorical + numerical are concatenated)
        x = Dense(units = 256)(x)
        #x = BatchNormalization()(x)
        x = Dropout(rate = 0.3)(x)
        x = ReLU()(x)
        x = Dropout(rate = 0.1)(x)
        x = Dense(128)(x)
        #x = BatchNormalization()(x)
        x = Dropout(rate = 0.3)(x)
        x = ReLU()(x)
        x = Dense(64)(x)
        #x = BatchNormalization()(x)
        x = Dropout(rate = 0.3)(x)
        x = ReLU()(x)
        x = Dense(32)(x)
        #x = BatchNormalization()(x)
        x = Dropout(rate = 0.3)(x)
        x = ReLU()(x)
        x = Dense(16)(x)
        #x = BatchNormalization()(x)
        x = Dropout(rate = 0.3)(x)
        x = ReLU()(x)

        output_layer = Dense(1, activation= 'sigmoid')(x) # ou sigmoid
        model_MLP = Model(inputs= [categorical_inputs ,numerical_input], outputs=output_layer) 
        model_MLP.summary()
        
        model_MLP.compile(optimizer= args.optimizer, loss=args.loss, metrics=['accuracy'])
        return model_MLP
        
    val_to_idx_mapping = {}
    for c in embed_cols:
        all_vals = np.unique(np.concatenate((X_train_resampled[c], X_test[c])))
        val_to_idx_mapping[c] = {val: idx for idx, val in enumerate(all_vals)}

    # Transform categorical columns using the value-to-index mapping
    input_list_train = []
    input_list_test = []
    for c in embed_cols:
        print("Unique values for", c, ":", col_vals_dict[c])
        print("Index mapping for", c, ":")
        for val, idx in val_to_idx_mapping[c].items():
            print(f"{val}: {idx}")
        print("\n")


    for c in embed_cols:
        input_list_train.append(np.array([val_to_idx_mapping[c][val] for val in X_train_resampled[c]]))
        input_list_test.append(np.array([val_to_idx_mapping[c][val] for val in X_test[c]]))

    # The rest of the columns (numerical features)
    numerical_cols = [c for c in X_train_resampled.columns if (not c in embed_cols)]
    input_list_train.append(X_train_resampled[numerical_cols].values)
    input_list_test.append(X_test[numerical_cols].values)


    batch_size = args.batch_size
    epochs = args.epochs
    print("fitting the model")
    NN = build_embedding_network(len_embed_cols)  #len(Cat_Var)
    hist_MLP = NN.fit(input_list_train, Y_train_resampled.values,batch_size=args.batch_size,
              epochs=args.epochs,
              shuffle=True, 
              validation_split = 0.1
    )
    plt.figure(figsize=(8, 6))
    plt.figure(1)
    plt.plot(hist_MLP.history['loss'], label='Train')
    plt.plot(hist_MLP.history['val_loss'], label='Validation')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.title('Loss_Curve')
    plt.savefig('loss_curve.png')
    mlflow.log_artifact("loss_curve.png")

    y_MLP = NN.predict(input_list_test)
    
    threshold = 0.5
    y_MLP_binary = (y_MLP > threshold).astype(int)

    accuracyscore = metrics.accuracy_score(Y_test,y_MLP_binary)*100
    f1score = metrics.f1_score(Y_test,y_MLP_binary)*100
    recallscore = metrics.recall_score(Y_test,y_MLP_binary)*100
    specificityscore = metrics.recall_score(Y_test, y_MLP_binary, pos_label=0)*100
    precisionscore = metrics.precision_score(Y_test,y_MLP_binary)*100


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

    # Visualize results
    cm = confusion_matrix(Y_test, y_MLP_binary)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    cm_display.figure_.savefig('confusion_matrix.png')
    mlflow.log_artifact("confusion_matrix.png")

    ######## COURBE DE ROC
    # calculate the fpr and tpr for all thresholds of the classification
    prob = y_MLP
    fpr_d, tpr_d, threshold = metrics.roc_curve(Y_test, prob)
    roc_auc = metrics.auc(fpr_d, tpr_d)
    print("AUC MLP=", roc_auc)
    plt.title('Courbe de ROC - MLP')
    plt.figure(2)
    plt.plot(fpr_d, tpr_d, 'b', label = 'AUC = %0.2f' % roc_auc)  # Using roc_auc directly
    plt.legend(loc='lower right')
    plt.figure(2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate - Sensitivity')
    plt.xlabel('False Positive Rate - (1-Specificity)')
    plt.savefig('roc_curve.png')
    mlflow.log_artifact("roc_curve.png")

    # Calcul de la courbe de précision-rappel
    precision, recall, _ = precision_recall_curve(Y_test, y_MLP)

    plt.figure(figsize=(8, 6))
    plt.figure(3)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.savefig('precision_recall.png')
    mlflow.log_artifact("precision_recall.png")
    plt.savefig('precision_recall_curve.png')
    mlflow.log_artifact("precision_recall_curve.png")

  
if __name__ == "__main__":
    
    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"input_data: {args.input_data}",
        f"model_output: {args.model_output}"
        f"units: {args.units}"
        f"learning_rate: {args.learning_rate}"
        f"activation: {args.activation}"
        f"optimizer: {args.optimizer}"
        f"loss: {args.loss}"
    ]
   
    
    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()





