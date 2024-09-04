# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:17:06 2022

@author: ASUS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# from sklearn.model_selection


# to store parameters
param_dict = {}

def missing_value_removal(df: pd.DataFrame, threshold=0.5, round_to=6) -> pd.DataFrame:
    """
    Missing value removal based on threshold and mean imputation

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame. Samples on index, features on columns
    threshold : float, optional
        Proportion threshold for feature removal. The default is 0.5.
    round_to : int, optional
        Decimal points of imputed values. The default is 6.

    Returns
    -------
    df : pd.DataFrame
        Imputed dataframe.

    """
    drop_list = []
    for col in df.columns:
        series = df[col]
        series_len = len(series)
        series_NaN = len([i for i in series.isna() if i == True])
        if series_NaN/series_len > threshold:
            drop_list.append(col)
        else:
            temp_series = series[[not i for i in series.isna()]]
            series_mean = round(np.mean(temp_series), round_to)
            series = series.fillna(series_mean)
            df[col] = series
    df.drop(drop_list, axis=1, inplace=True)
    return df


def pca_df(df):
    scaled_df = preprocessing.scale(df.T)
    scaled_df.shape
    pca = PCA()
    pca.fit(scaled_df)
    pca_data = pca.transform(scaled_df)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    indexes = df.index
    heads = list(df.columns)
    pca_df = pd.DataFrame(pca_data, index=heads, columns=labels)
    return pca_df, per_var


def class_weight_gen(y, weight_grid):
    """    Class weight optimisation for imbalanced dataset for GridSearchCV.    
    Uses compute_class_weight utility from sklearn.utils to generate    
    balanced class weight for BINARY classification.    
    Useful for datasets that cannot solve imbalanced dataset issue with    
    class_weight = 'balanced' parameter.        
    Uses exponents (> 0) to increase/decrease the distance of the weights.        
    More than 1 increases the distance of the weights while     
    Less than 1 decreases the distance of the weights.            
    Parameters    
    ----------    
    y : iterable (list, 1-d array, series)        
    Categorical labels, str and int compatible    
    weight_grid : iterable (list, tuple)        
    List of power factors to optimise for.    
    Returns    
    -------    
    cw_grid : list of dictionary objects        
    List of dictionary objects with modified weights according to grid    
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.sort(np.unique(y)) # get classes for labelling     
    # class weight optimisation    
    weights = compute_class_weight(class_weight = 'balanced', classes = classes, y =  y)
    fx = lambda x, y: x**y    
    cw_grid = [dict(zip(classes,[fx(i, j) for i in weights])) for j in weight_grid]
    return cw_grid
# loading feature sets selected from feature selection
hcl_scz_feature_set = ['CD8A', 'MCP-3', 'IL-17C', 'MCP-1', 'IL-17A', 'CST5', 'CD6', 'SCF', 'MCP-4', 'IL-15RA', 'IL-10RB', 'IL-12B', 'MMP-10', 'IL10', 'CCL23', 'CD5', 'CXCL10', 'DNER']
are_art_feature_set = ['CST5', 'MCP-4', 'IL-18R1', 'PD-L1', 'TNF', 'MCP-2', 'CCL25','TNFRSF9']
cre_crt_feature_set = ['MCP-3', 'IL-17A', 'AXIN1', 'CXCL1', 'FGF-23', 'CCL19', 'IL10','TNF', 'FGF-19', 'MCP-2', 'CCL25']


# loading datasets
df1 = pd.read_csv('part1_full_df.csv', index_col=0)
df2 = pd.read_csv('part2_full_df.csv', index_col=0)
df3 = pd.read_csv('part3_full_df.csv', index_col=0)


# full metadata
meta = pd.read_csv('csa_meta.csv', index_col=0)
meta.columns = ['Class', 'Gender', 'Ethnicity', 
                'BMI', 'Age', 'RNA-Seq Batch']
meta = meta[meta.Class != 'Terminated']


# abbreviating class labels
temp = {'Healthy control': 'HCL',
        'Clozapine resistant': 'CRT',
        'Clozapine responsive': 'CRE',
        'Antipsychotic responsive': 'ARE'}


meta['cl'] = [temp[i] for i in meta.Class]

# encoding binary labels
for cl in meta.cl.unique():
    cl_n = cl
    meta[cl_n] = [cl if meta.loc[i, 'cl'] ==
                  cl else 'NOT_%s' % (cl) for i in meta.index]

# MV filtering and mean imputation
df1 = missing_value_removal(df1)
df2 = missing_value_removal(df2)
df3 = missing_value_removal(df3)

# dropping non shared genes
shared_genes = [i for i in df1.columns if (i in df2.columns) and i in (df3.columns)]
df1 = df1.loc[:, shared_genes]
df2 = df2.loc[:, shared_genes]
df3 = df3.loc[:, shared_genes]


# identifying shared samples for reference
shared_samples = [i for i in df1.index if (i in df2.index) or (i in df3.index)]


# adjustment function for median centering
def temp(x, y): return [x[i] - y[i] for i in range(len(x))]

# adjustment factor = median of difference between shared samples of each gene
for gene in shared_genes:
    normalisation_factor = np.median(temp(df1.loc[shared_samples, gene],
                                          df2.loc[shared_samples, gene]))

    def normalise(x): return x + normalisation_factor
    df2.loc[:, gene] = df2.loc[:, gene].map(normalise)


shared_samples = [i for i in df1.index if i in df3.index]

def temp(x, y): return [x[i] - y[i] for i in range(len(x))]


for gene in shared_genes:
    normalisation_factor = np.median(temp(df1.loc[shared_samples, gene],
                                          df3.loc[shared_samples, gene]))

    def normalise(x): return x + normalisation_factor
    df3.loc[:, gene] = df3.loc[:, gene].map(normalise)

# deep copy 
train_df = df1.copy()
val_df = df2.copy()
test_df = df3.copy()

# removing shared samples from test set to prevent data leakage
test_df = test_df.loc[[i for i in test_df.index if i not in shared_samples]]
train_meta = meta.loc[train_df.index]
val_meta = meta.loc[val_df.index]
test_meta = meta.loc[test_df.index]

train_meta.sort_values('Class', inplace = True)
val_meta.sort_values('Class', inplace = True)
test_meta.sort_values('Class', inplace = True)

# parameter grid for gridsearch
param_grid = {'C': [i/100 for i in range(1,100)],
              'gamma': ['scale'],
              'decision_function_shape': ['ovo'],
              'kernel': ['linear','rbf'],
              'max_iter': [-1]}


# HCL/SCZ
lut = {'HCL': 'HCL',
       'NOT_HCL': 'SCZ'}

label = 'HCL'
features = hcl_scz_feature_set

# making sure the index of datafreame match the class label
train_df = train_df.loc[train_meta.index]
val_df = val_df.loc[val_meta.index]
test_df = test_df.loc[test_meta.index]

# defining datasets
X_train, X_val, X_test = train_df, val_df, test_df
y_train, y_val, y_test = train_meta.loc[:,label].map(lut), val_meta.loc[:,label].map(lut), test_meta.loc[:,label].map(lut)


# loading selected features
X_train, X_val, X_test = X_train.loc[:,features], X_val.loc[:,features], X_test.loc[:,features]

# merging training set and validation set for cross validation

X_train.index = [f'{i}_1' for i in X_train.index]
X_val.index = [f'{i}_2' for i in X_val.index]
y_train.index = [f'{i}_1' for i in y_train.index]
y_val.index = [f'{i}_2' for i in y_val.index]


y_train = pd.concat([y_train, y_val])
X_train = pd.concat([X_train, X_val]).loc[y_train.index, features]

# setting up range of class weights to be sampled
cw = class_weight_gen(y_train, np.linspace(-2,2,50))
param_grid['class_weight'] = cw


cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
svc = SVC(probability = True)
clf = GridSearchCV(svc,
                   scoring = 'balanced_accuracy',
                   n_jobs = -1,
                   cv = cv,
                   param_grid = param_grid)


# fit model
clf.fit(X_train, y_train)

# save best parameter
param_dict[label] = clf.best_params_

# test using test set
X_test = X_test.loc[:,features]
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)

metrics.roc_auc_score(y_test.tolist(), y_score[:,1])
metrics.confusion_matrix(y_test.tolist(), y_pred)



# loading plots (ROC curve and confusion matrix)
fpr, tpr, threshold = metrics.roc_curve(y_test.factorize()[0], y_score[:,0])
roc_auc_score = metrics.roc_auc_score(y_test.factorize()[0], y_score[:,0])

fig, ax = plt.subplots(1,1,dpi = 150, figsize = (4,3))
a = metrics.RocCurveDisplay(fpr = fpr, tpr = tpr, roc_auc = roc_auc_score)
a.plot(ax = ax, color = 'k')
sns.lineplot(x = [0,1], y = [0,1], color = 'red', linestyle = ':')
bal_acc = metrics.balanced_accuracy_score(y_test, y_pred)
fig.suptitle(f'Test set AUC for {label}, balanced acc: {bal_acc:.2f}')

lut = list(lut.values())

plt.figure(dpi = 150, figsize = (3,2.5))
sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot = True, 
            xticklabels = [lut[0], lut[1]],
            yticklabels = [lut[0], lut[1]],
            cmap = 'viridis')
plt.title('Confusion matrix for test prediction')




# ARE/ART
lut = {'ARE': 'ARE',
       'NOT_ARE': 'ART'}

label = 'ARE'

# removing HCL for antipsychotics response classification
train_meta = train_meta[train_meta.HCL != 'HCL']
val_meta = val_meta[val_meta.HCL != 'HCL']
test_meta = test_meta[test_meta.HCL != 'HCL']

# loading are/art feature set
features = are_art_feature_set

# making sure the index of datafreame match the class label
train_df = train_df.loc[train_meta.index]
val_df = val_df.loc[val_meta.index]
test_df = test_df.loc[test_meta.index]

# defining datasets
X_train, X_val, X_test = train_df, val_df, test_df
y_train, y_val, y_test = train_meta.loc[:,label].map(lut), val_meta.loc[:,label].map(lut), test_meta.loc[:,label].map(lut)


# loading selected features
X_train, X_val, X_test = X_train.loc[:,features], X_val.loc[:,features], X_test.loc[:,features]



# setting up range of class weights to be sampled
cw = class_weight_gen(y_train, np.linspace(-2,2,50))
param_grid['class_weight'] = cw


cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
svc = SVC(probability = True)
clf = GridSearchCV(svc,
                   scoring = 'balanced_accuracy',
                   n_jobs = -1,
                   cv = cv,
                   param_grid = param_grid)


# merging training set and validation set for cross validation

X_train.index = [f'{i}_1' for i in X_train.index]
X_val.index = [f'{i}_2' for i in X_val.index]
y_train.index = [f'{i}_1' for i in y_train.index]
y_val.index = [f'{i}_2' for i in y_val.index]


y_train = pd.concat([y_train, y_val])
X_train = pd.concat([X_train, X_val]).loc[y_train.index, features]

# fit model
clf.fit(X_train, y_train)

# save best parameter
param_dict[label] = clf.best_params_

# test using test set
X_test = X_test.loc[:,features]
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)

metrics.roc_auc_score(y_test.tolist(), y_score[:,1])
metrics.confusion_matrix(y_test.tolist(), y_pred)



# loading plots (ROC curve and confusion matrix)
fpr, tpr, threshold = metrics.roc_curve(y_test.factorize()[0], y_score[:,0])
roc_auc_score = metrics.roc_auc_score(y_test.factorize()[0], y_score[:,0])

fig, ax = plt.subplots(1,1,dpi = 150, figsize = (4,3))
a = metrics.RocCurveDisplay(fpr = fpr, tpr = tpr, roc_auc = roc_auc_score)
a.plot(ax = ax, color = 'k')
sns.lineplot(x = [0,1], y = [0,1], color = 'red', linestyle = ':')
bal_acc = metrics.balanced_accuracy_score(y_test, y_pred)
fig.suptitle(f'Test set AUC for {label}, balanced acc: {bal_acc:.2f}')

lut = list(lut.values())

plt.figure(dpi = 150, figsize = (3,2.5))
sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot = True, 
            xticklabels = [lut[0], lut[1]],
            yticklabels = [lut[0], lut[1]],
            cmap = 'viridis')
plt.title('Confusion matrix for test prediction')



# CRE/CRT

lut = {'CRE': 'CRE',
       'NOT_CRE': 'CRT'}

label = 'CRE'

train_meta = train_meta[train_meta.HCL != 'HCL']
val_meta = val_meta[val_meta.HCL != 'HCL']
test_meta = test_meta[test_meta.HCL != 'HCL']

train_meta = train_meta[train_meta.ARE != 'ARE']
val_meta = val_meta[val_meta.ARE != 'ARE']
test_meta = test_meta[test_meta.ARE != 'ARE']



# loading are/art feature set
features = cre_crt_feature_set

# making sure the index of datafreame match the class label
train_df = train_df.loc[train_meta.index]
val_df = val_df.loc[val_meta.index]
test_df = test_df.loc[test_meta.index]

# defining datasets
X_train, X_val, X_test = train_df, val_df, test_df
y_train, y_val, y_test = train_meta.loc[:,label].map(lut), val_meta.loc[:,label].map(lut), test_meta.loc[:,label].map(lut)


# loading selected features
X_train, X_val, X_test = X_train.loc[:,features], X_val.loc[:,features], X_test.loc[:,features]



# setting up range of class weights to be sampled
cw = class_weight_gen(y_train, np.linspace(-2,2,50))
param_grid['class_weight'] = cw


cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
svc = SVC(probability = True)
clf = GridSearchCV(svc,
                   scoring = 'balanced_accuracy',
                   n_jobs = -1,
                   cv = cv,
                   param_grid = param_grid)


# merging training set and validation set for cross validation

X_train.index = [f'{i}_1' for i in X_train.index]
X_val.index = [f'{i}_2' for i in X_val.index]
y_train.index = [f'{i}_1' for i in y_train.index]
y_val.index = [f'{i}_2' for i in y_val.index]


y_train = pd.concat([y_train, y_val])
X_train = pd.concat([X_train, X_val]).loc[y_train.index, features]

# fit model
clf.fit(X_train, y_train)

# save best parameter
param_dict[label] = clf.best_params_

# test using test set
X_test = X_test.loc[:,features]
y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)

metrics.roc_auc_score(y_test.tolist(), y_score[:,1])
metrics.confusion_matrix(y_test.tolist(), y_pred)



# loading plots (ROC curve and confusion matrix)
fpr, tpr, threshold = metrics.roc_curve(y_test.factorize()[0], y_score[:,0])
roc_auc_score = metrics.roc_auc_score(y_test.factorize()[0], y_score[:,0])

fig, ax = plt.subplots(1,1,dpi = 150, figsize = (4,3))
a = metrics.RocCurveDisplay(fpr = fpr, tpr = tpr, roc_auc = roc_auc_score)
a.plot(ax = ax, color = 'k')
sns.lineplot(x = [0,1], y = [0,1], color = 'red', linestyle = ':')
bal_acc = metrics.balanced_accuracy_score(y_test, y_pred)
fig.suptitle(f'Test set AUC for {label}, balanced acc: {bal_acc:.2f}')

lut = list(lut.values())

plt.figure(dpi = 150, figsize = (3,2.5))
sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot = True, 
            xticklabels = [lut[0], lut[1]],
            yticklabels = [lut[0], lut[1]],
            cmap = 'viridis')
plt.title('Confusion matrix for test prediction')

