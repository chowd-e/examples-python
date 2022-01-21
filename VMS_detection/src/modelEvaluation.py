import os, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import RocCurveDisplay

from xgboost import XGBClassifier, XGBRegressor
from scipy.spatial.distance import pdist, squareform

# read csv data storing classification data for test / train from VMS output
def load(path):
   # Read / Parse CSV with conversion of dates to datetime
   data = pd.read_csv(path)
   scaler = MinMaxScaler(feature_range = (0, 1)) # don't leave zeroes
   npData = scaler.fit_transform(data)
   data = pd.DataFrame(npData, columns = data.columns)

   cumNA = []
   for colname in data.columns:
      naMask = data[colname].isna()
      cumNA.append((sum(naMask), naMask[naMask == 1].index[0].tolist()))

   return data, cumNA

# calculate and display the feature distributions across teh data, save at 
# path_wr
def dist(data, path_wr):
   try:
      os.chdir(path_wr)
      for colname in data.columns:
         fig = plt.figure(figsize = (5,5))
         sns.set()
         sns.distplot(data[colname], hist = True, kde = False)
         plt.title("Histogram: {colname}")
         plt.savefig(str("dist_" + colname), format = 'png' )
      
      # return a series of figure? or parameterize path to save?
      return True
   except:
      return False

# calculate correlation matrix between features, return figure
def correlation(data):
   corr = data.corr()
   
   # mask to show only upper half of triangular array
   corrMask = np.zeros_like(corr, dtype = np.bool) 

   # upper triangle converted to TRUE
   corrMask[np.triu_indices_from(corrMask)] = True 

   # general plot formatting
   sns.set_style(style = 'ticks')
   sns.set_context("paper")
   fig, ax = plt.subplots(figsize = (15, 12))

   # plot params
   sns.heatmap(corr, 
               mask = corrMask, 
               cmap = 'YlGnBu', 
               annot = True,
               square = True, 
               linewidths = 0.5, 
               cbar_kws = {"shrink": 0.75},
               ax = ax)
   return fig

# compare set of classifiers (Logistic Regression, Decision Tree, Naive Bayes, 
# Random Forest and XGBoost)
# Plot ROC curve and print dataframe of metrics
def evalClassifiers(data):
   xtrain, xtest, ytrain, ytest = splitData(data)

   models = {} # container for (name, model) tuple
   
   # Setting Models
   lr = LogisticRegression(solver = 'lbfgs', max_iter=10000)
   dt = DecisionTreeClassifier()
   nb = GaussianNB()
   rfc = RandomForestClassifier()
   bst = XGBClassifier(objective = 'binary:logistic', use_label_encoder = False, eval_metric = 'logloss')

   models = {'Logistic Regression': lr, 
             'Decision Tree': dt, 
             'Naive Bayes': nb, 
             'Random Forest' : rfc, 
             'XGBoost' : bst
            }

   # new figure
   figure = plt.figure()
      
   # consistent folding parameters accross validation
   kfold = KFold(n_splits = 10) 

   # set of metrics to assess
   metrics = ['precision', 'recall', 'f1', 'roc_auc'] 

   # Precision, Recall, F-Score, AUC, and ROC curve plot
   for name, model in models.items():
      print(name + ":")
      
      model.fit(xtrain, ytrain.values.ravel())
      for metric in metrics:
         score = cross_val_score(model, xtest, ytest.values.ravel(), cv = kfold, scoring = metric)
         print("\t" + str(metric) + ": %0.2f, std: %0.2f" % (score.mean(), score.std()))
         
      ax = plt.gca() # add to current axis or create axis
      disp = RocCurveDisplay.from_estimator(model, xtest, ytest, ax = ax, alpha = 0.8)

   return figure

# split input data into X and Y data, if not targetName given, assume last col
def getXY(data, targetName = None):
   if targetName == None:
      targetName = data.iloc[:, len(data.columns) - 1].name 
   
   targetMask = data.columns == targetName
   X = data.loc[:,~targetMask]
   Y = data.loc[:, targetMask]

   return X, Y

# Split Data into test and train sets for X and Y
def splitData(data):
   X, Y = getXY(data)

   # split into test train sections
   xtrain, xtest, ytrain, ytest = train_test_split(X,
                                                   Y,
                                                   test_size = 0.33, 
                                                   random_state = 32)

   return xtrain, xtest, ytrain, ytest

# Evalaute regressio models
def evalRegressors(data):
   xtrain, xtest, ytrain, ytest = splitData(data)
   
   lr = LogisticRegression()
   bst = XGBRegressor()
   models = {'LogisticRegressor':lr, 'XGBoost':bst}
   # new figure
   figure = plt.figure()
      
   # consistent folding parameters accross validation
   kfold = KFold(n_splits = 10) 

   # set of metrics to assess
   metrics = ['neg_mean_absolute_error', 
               'neg_mean_squared_error', 
               'neg_root_mean_squared_error', 
               'neg_median_absolute_error'
             ] 

   # Precision, Recall, F-Score, AUC, and ROC curve plot
   for name, model in models.items():
      print(name + ":")
      
      model.fit(xtrain, ytrain.values.ravel())
      for metric in metrics:
         score = cross_val_score(model, xtest, ytest.values.ravel(), cv = kfold, scoring = metric)
         print("\t" + str(metric) + ": %0.2f, std: %0.2f" % (score.mean(), score.std()))

# Visualize distance in feature space via cluster analysis TNSE
def clusterVisualization(data, labelVals = [0,1]):
    lenlist=[0]
    _,Y = getXY(data)
    targetName = Y.columns.values[0]
    
    data_sub = data[data[targetName] == labelVals[0]]
    lenlist.append(data_sub.shape[0])
    
    # Iterate through the sublist concatenating indexed values
    for label in labelVals[1:]:
        temp = data[data[targetName] == label]
        data_sub = pd.concat([data_sub, temp],axis=0,ignore_index=True)
        lenlist.append(data_sub.shape[0])

    # Drop target variables to avoid plotting
    df_X = data_sub.drop(targetName,axis=1)
    dist = squareform(pdist(df_X, metric='cosine'))  
    tsne = TSNE(n_components = 3,
                init = 'random',
                perplexity = 35,
                learning_rate = 200,
                random_state = 42, 
                square_distances = True, 
                metric='precomputed').fit_transform(dist)

    palette = sns.color_palette("hls", len(labelVals))
    plt.figure(figsize=(10,10))

    for i,cuisine in enumerate(labelVals):
        plt.scatter(tsne[lenlist[i]:lenlist[i+1],0],\
        tsne[lenlist[i]:lenlist[i+1],1],c=palette[i],label=labelVals[i])

    plt.legend()
    plt.show()

# save / eport the model to be loaded at separate time
def exportModel(model, filename = 'model.pkl'):
   try:
      # validate file extention
      if ~filename.endswith('.pkl'):
         filename += ".pkl"

      with open(filename, 'wb') as f:
         pickle.dump(model, f)
   except:
      return -1

   return 0

# load pickled model and return
def loadModel(path):
   try:
      with open(path, 'rb') as f:
         model = pickle.load(f)
   except:
      return None

   return model