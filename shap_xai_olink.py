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

full_metric = {}

param_dict = {'HCL': {'C': 1.0481131341546852,
              'class_weight': {'HCL': 2.1766618793992865, 'SCZ': 0.6321956810424791},
              'gamma': 0.1,
              'kernel': 'rbf',
              'max_iter': -1},
             'ARE': {'C': 0.84,
              'class_weight': {'ARE': 1.5841439252774316, 'ART': 0.7186019007635811},
              'decision_function_shape': 'ovo',
              'gamma': 'scale',
              'kernel': 'linear',
              'max_iter': -1},
             'CRE': {'C': 0.73,
              'class_weight': {'CRE': 0.9759097652299161, 'CRT': 1.0240524419089565},
              'decision_function_shape': 'ovo',
              'gamma': 'scale',
              'kernel': 'linear',
              'max_iter': -1}}
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
shared_samples = [i for i in df1.index if (i in df2.index) and (i in df3.index)]


# adjustment function for median centering
def temp(x, y): return [x[i] - y[i] for i in range(len(x))]


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


train_df = df1.copy()
val_df = df2.copy()
test_df = df3.copy()

test_df = test_df.loc[[i for i in test_df.index if i not in shared_samples]]
train_meta = meta.loc[train_df.index]
val_meta = meta.loc[val_df.index]
test_meta = meta.loc[test_df.index]

train_meta.sort_values('Class', inplace = True)
val_meta.sort_values('Class', inplace = True)
test_meta.sort_values('Class', inplace = True)

# lookup table for plot titles.
waterfall_title = {"HCL":"HCL/SCZ",
                   "ARE":"ARE/ART",
                   "CRE":"CRE/CRT"}

# HCL/SCZ

lut = {'HCL': 'HCL',
       'NOT_HCL': 'SCZ'}

label = 'HCL'

# matching matrix index to class label index
train_df = train_df.loc[train_meta.index]
val_df = val_df.loc[val_meta.index]
test_df = test_df.loc[test_meta.index]

# setting into train, val and test format
X_train, X_val, X_test = train_df, val_df, test_df
y_train, y_val, y_test = train_meta.loc[:,label].map(lut), val_meta.loc[:,label].map(lut), test_meta.loc[:,label].map(lut)

# loading features
features = hcl_scz_feature_set
X_train, X_val, X_test = X_train.loc[:,features], X_val.loc[:,features], X_test.loc[:,features]


X_train, X_val, X_test = train_df.loc[:,features], val_df.loc[:,features], test_df.loc[:,features]
y_train, y_val, y_test = train_meta.loc[:,label].map(lut), val_meta.loc[:,label].map(lut), test_meta.loc[:,label].map(lut)

X_train.index = [f'{i}_1' for i in X_train.index]
X_val.index = [f'{i}_2' for i in X_val.index]
y_train.index = [f'{i}_1' for i in y_train.index]
y_val.index = [f'{i}_2' for i in y_val.index]

y_train = pd.concat([y_train, y_val])
X_train = pd.concat([X_train, X_val]).loc[y_train.index, features]

# loading parameters from modeling
clf = SVC(probability = True)
clf.set_params(**param_dict[label])
clf.fit(X_train, y_train)


import matplotlib.pyplot as plt
import shap

# explainable results extracted from test set
y_pred = clf.predict(X_test)

status = ['Classified' if (i == y_test[ind]) else 'Misclassified' for ind, i in enumerate(y_pred)]
y_true = y_test.tolist()

explainer = shap.KernelExplainer(clf.predict_proba, X_test)
shap_values = explainer.shap_values(X_test)
feature_names = X_train.columns.tolist()

# Create a custom summary plot
fig, ax = plt.subplots(figsize=(10, 7), dpi = 150)

# Use SHAP's built-in function but capture its axes output
# Note: We're using shap_values[1] because we're explaining class 1
shap.summary_plot(shap_values[:,:,1], X_test, feature_names=feature_names, plot_type="bar", show=False, color="dodgerblue")


# Enhance aesthetics for publication
ax.set_title("Feature Importance using SHAP values (SVM) - " + label, fontsize=16)
ax.set_xlabel("Mean Absolute SHAP Value", fontsize=14)
ax.set_ylabel("Features", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()


# ARE/ART
lut = {'ARE': 'ARE',
       'NOT_ARE': 'ART'}

label = 'ARE'

train_meta = train_meta[train_meta.HCL != 'HCL']
val_meta = val_meta[val_meta.HCL != 'HCL']
test_meta = test_meta[test_meta.HCL != 'HCL']


train_df = train_df.loc[train_meta.index]
val_df = val_df.loc[val_meta.index]
test_df = test_df.loc[test_meta.index]


X_train, X_val, X_test = train_df, val_df, test_df
y_train, y_val, y_test = train_meta.loc[:,label].map(lut), val_meta.loc[:,label].map(lut), test_meta.loc[:,label].map(lut)


features = are_art_feature_set
X_train, X_val, X_test = X_train.loc[:,features], X_val.loc[:,features], X_test.loc[:,features]

clf = SVC(probability = True)

X_train, X_val, X_test = train_df.loc[:,features], val_df.loc[:,features], test_df.loc[:,features]
y_train, y_val, y_test = train_meta.loc[:,label].map(lut), val_meta.loc[:,label].map(lut), test_meta.loc[:,label].map(lut)

X_train.index = [f'{i}_1' for i in X_train.index]
X_val.index = [f'{i}_2' for i in X_val.index]
y_train.index = [f'{i}_1' for i in y_train.index]
y_val.index = [f'{i}_2' for i in y_val.index]

y_train = pd.concat([y_train, y_val])
X_train = pd.concat([X_train, X_val]).loc[y_train.index, features]

clf.set_params(**param_dict[label])
clf.fit(X_train, y_train)


import matplotlib.pyplot as plt
import shap

y_pred = clf.predict(X_test)

# identify reason for classified and misclassified
status = ['Classified' if (i == y_test[ind]) else 'Misclassified' for ind, i in enumerate(y_pred)]
y_true = y_test.tolist()

explainer = shap.KernelExplainer(clf.predict_proba, X_test)
shap_values = explainer.shap_values(X_test)
feature_names = X_train.columns.tolist()

# create summary plot
fig, ax = plt.subplots(figsize=(10, 7), dpi = 150)

# create summary plot
shap.summary_plot(shap_values[:,:,1], X_test, feature_names=feature_names, plot_type="bar", show=False, color="dodgerblue")

# Enhance aesthetics 
ax.set_title("Feature Importance using SHAP values (SVM) - " + label, fontsize=16)
ax.set_xlabel("Mean Absolute SHAP Value", fontsize=14)
ax.set_ylabel("Features", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig(f'{label}_summary_plot.jpg', dpi = 300)

plt.show()


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


train_df = train_df.loc[train_meta.index]
val_df = val_df.loc[val_meta.index]
test_df = test_df.loc[test_meta.index]


X_train, X_val, X_test = train_df, val_df, test_df
y_train, y_val, y_test = train_meta.loc[:,label].map(lut), val_meta.loc[:,label].map(lut), test_meta.loc[:,label].map(lut)


features = cre_crt_feature_set
X_train, X_val, X_test = X_train.loc[:,features], X_val.loc[:,features], X_test.loc[:,features]

clf = SVC(probability = True)

X_train, X_val, X_test = train_df.loc[:,features], val_df.loc[:,features], test_df.loc[:,features]
y_train, y_val, y_test = train_meta.loc[:,label].map(lut), val_meta.loc[:,label].map(lut), test_meta.loc[:,label].map(lut)

X_train.index = [f'{i}_1' for i in X_train.index]
X_val.index = [f'{i}_2' for i in X_val.index]
y_train.index = [f'{i}_1' for i in y_train.index]
y_val.index = [f'{i}_2' for i in y_val.index]

y_train = pd.concat([y_train, y_val])
X_train = pd.concat([X_train, X_val]).loc[y_train.index, features]

clf.set_params(**param_dict[label])
clf.fit(X_train, y_train)

# workflow from modelling above.
 
import matplotlib.pyplot as plt
import shap

y_pred = clf.predict(X_test)

status = ['Classified' if (i == y_test[ind]) else 'Misclassified' for ind, i in enumerate(y_pred)]
y_true = y_test.tolist()

explainer = shap.KernelExplainer(clf.predict_proba, X_test)
shap_values = explainer.shap_values(X_test)
feature_names = X_train.columns.tolist()


# create a custom summary plot
fig, ax = plt.subplots(figsize=(10, 7), dpi = 150)


shap.summary_plot(shap_values[:,:,1], X_test, feature_names=feature_names, plot_type="bar", show=False, color="dodgerblue")

# Enhance aesthetics 
ax.set_title("Feature Importance using SHAP values (SVM) - " + label, fontsize=16)
ax.set_xlabel("Mean Absolute SHAP Value", fontsize=14)
ax.set_ylabel("Features", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig(f'{label}_summary_plot.jpg', dpi = 300)

