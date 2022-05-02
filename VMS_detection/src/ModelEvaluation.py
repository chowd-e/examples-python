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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from xgboost import XGBClassifier, XGBRegressor
from scipy.spatial.distance import pdist, squareform

models ={'Logistic_Regression': LogisticRegression(solver = 'lbfgs'), 
         'Decision_Tree': DecisionTreeClassifier(), 
         'Naive_Bayes': GaussianNB(), 
         'Random_Forest' : RandomForestClassifier(), 
         'XGBoost' : XGBClassifier(objective='binary:logistic',
                                   use_label_encoder = False, 
                                   eval_metric = 'logloss')
         }

metrics = {'classifier':['accuracy', 
                         'precision', 
                         'recall', 
                         'f1',
                         'roc_auc', 
                        ],
            'regressor':['neg_mean_absolute_error', 
                         'neg_mean_squared_error', 
                         'neg_root_mean_squared_error', 
                         'neg_median_absolute_error',
                        ],
}

def getModelNames():
   return models.keys()

# read csv data storing classification data for test / train from VMS output
def loadFeatures(path):
   # Read / Parse CSV with conversion of dates to datetime
   data = pd.read_csv(path)
   scaler = MinMaxScaler(feature_range = (0, 1)) # don't leave zeroes
   npData = scaler.fit_transform(data)
   data = pd.DataFrame(npData, columns = data.columns)
   data.dropna(inplace=True)
   
   return data, scaler

# calculate and display the feature distributions across teh data, save at 
# path_wr
def getFeatureDistribution(data, path_wr = None):
   try:
      os.chdir(path_wr)
      for colname in data.columns:
         fig = plt.figure(figsize = (5,5))
         sns.set()
         sns.distplot(data[colname], hist = True, kde = False)
         plt.title("Histogram: {colname}")
         plt.savefig("dist_{colname}", format = 'png' )
      
      # return a series of figure? or parameterize path to save?
      return 0
   except:
      return -1

# calculate correlation matrix between features, return figure
def getFeatureCorrelation(data):
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
def evaluateClassifiers(data, model_names = []):
   xtrain, xtest, ytrain, ytest = splitData(data)

   if model_names == []:
      mods = getModelNames()

   # new figure
   figure = plt.figure()
      
   # consistent folding parameters accross validation
   kfold = KFold(n_splits = 10) 

   # set of metrics to assess
   metrics = ['precision', 'recall', 'f1', 'roc_auc'] 

   # Precision, Recall, F-Score, AUC, and ROC curve plot
   for name, model in mods.items():
      print(name + ":")
      
      model.fit(xtrain, ytrain.values.ravel())
      for metric in metrics:
         score = cross_val_score(model, xtest, ytest.values.ravel(), cv = kfold, scoring = metric)
         print("\t" + str(metric) + ": %0.2f, std: %0.2f" % (score.mean(), score.std()))
         
      ax = plt.gca() # add to current axis or create axis
      disp = RocCurveDisplay.from_estimator(model, xtest, ytest, ax = ax, alpha = 0.8)

   return figure

def getTargetVariable(data):
   return data.iloc[:, len(data.columns) - 1].name

# split input data into X and Y data, if not targetName given, assume last col
def getXY(data, targetName = None):
   if targetName == None:
      targetName = getTargetVariable(data) 
   
   targetMask = data.columns == targetName
   x = data.loc[:,~targetMask]
   y = data.loc[:, targetMask]

   return x, y

# Split Data into test and train sets for X and Y
def splitData(data):
   x, y = getXY(data)

   # split into test train sections
   xtrain, xtest, ytrain, ytest = train_test_split(x,
                                                   y,
                                                   test_size = 0.33, 
                                                   random_state = 42)

   return xtrain, xtest, ytrain, ytest

def get_test_data(data):
   x, _, y, _ = splitData(data)
   return x, y

def get_train_data(data):
   _, x, _, y = splitData(data)
   return x, y

# Evalaute regressio models
def evaluateRegressors(data):
   xtrain, xtest, ytrain, ytest = splitData(data)
   
   lr = LogisticRegression()
   bst = XGBRegressor()
   models = {'Logistic_Regressor':lr, 'XGBoost':bst}
   # new figure
   fig = plt.figure()
      
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
def showTNSE(data, labelVals = [0,1]):
    lenlist=[0]

    targetName = getTargetVariable(data)
    
    data_sub = data[data[targetName] == labelVals[0]]
    lenlist.append(data_sub.shape[0])
    
    # Iterate through the sublist concatenating indexed values
    for label in labelVals[1:]:
        temp = data[data[targetName] == label]
        data_sub = pd.concat([data_sub, temp],
                              axis=0,
                              ignore_index=True)
        lenlist.append(data_sub.shape[0])

    # Drop target variables to avoid plotting
    df_X = data_sub.drop(targetName, axis=1)
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

    for i in enumerate(labelVals):
        plt.scatter(tsne[lenlist[i]:lenlist[i+1],0],
                    tsne[lenlist[i]:lenlist[i+1],1],
                    c = palette[i],
                    label = labelVals[i])

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
def importModel(path):
   try:
      with open(path, 'rb') as f:
         model = pickle.load(f)
   except:
      return None

   return model

def trainModel(data, model_selection = 'Decision_Tree'): 
   xtrain, ytrain = get_train_data(data)

   if model_selection not in models:
      return None

   model = models[model_selection]
   model.fit(xtrain, ytrain.values.ravel())
   return model

def predict(data, trained_model):
   x, _ = getXY(data)
   preds = trained_model.predict(x)
   return pd.DataFrame(preds, columns = ["preds"])

def showConfusionMatrix(data, trained_model):
   x, _ = getXY(data)
   ypreds = trained_model.predict(x)
   ypreds = pd.DataFrame(ypreds, columns = ["preds"])
   targetVar = getTargetVariable(data) 

   # build confusion matrix
   conf_mat = confusion_matrix(ypreds, data[targetVar])
   fig = ConfusionMatrixDisplay(confusion_matrix = conf_mat)
   fig.plot()
   plt.title("Confusion Matrix")  
   plt.show()
   return fig

def getMetricList(model_type = 'classifier'):
   if model_type in metrics:
      return metrics[model_type]
   else:
      return None

def computeMetrics(data, trained_model, model_type='classifier'):
   metrics = getMetricList(model_type)
   output = pd.DataFrame(columns=metrics)
   row = {}
   x, y = getXY(data)
   # xtest, ytest = get_test_data(data)
   # consistent folding parameters accross validation
   kfold = KFold(n_splits = 10, shuffle=True, random_state=42)
   for metric in metrics:
      score = cross_val_score(trained_model, 
                              x, 
                              y.values.ravel(), 
                              cv = kfold, 
                              scoring = metric)
      
      row[metric] = score.mean()
   return output.append(row, ignore_index=True)

def plotTable(df, title = 'Table Display', ax=None):
   if not isinstance(df, pd.DataFrame):
      return -1

   ccolors = plt.cm.BuPu(np.full(len(df.columns), 0.1))
   
   if ax == None:
      fig, ax = plt.subplots()
      fig.patch.set_visible(False)
   
   ax.axis('off')
   ax.axis('tight')
   ax.set_title(title, fontweight='bold')

   # ax.set_axis_off()
   table = ax.table(
      cellText=np.round(df.values, 4),
      colLabels=df.columns,
      colColours=ccolors,
      loc='upper left',
      cellLoc='center',
   )

   table.auto_set_font_size(False)
   table.set_fontsize(10)
   # table.scale(2,2)
   # fig.tight_layout()
   return ax

if __name__ == '__main__':
   path_in = "C:/git/python-examples/VMS_detection/data/VMS_dataset.csv"
   data, _ = loadFeatures(path_in)
   model = trainModel(data, selection='RandomForest')
   conf = showConfusionMatrix(data, model)
   metric = computeMetrics(data, model)
   print(metric)




