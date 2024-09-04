# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:17:06 2022

@author: Ser-Xian Phua
"""

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

def svm_rfe_feature_selection(X_train,
                              y_train,
                              X_test,
                              y_test,
                              param_grid=None,
                              min_features=5, 
                              reduction_factor=0.9,
                              random_state=42):
    """
    

    Parameters
    ----------
    X_train : pandas df training matrix
    y_train : pandas series training labels
    X_test : pandas df validation matrix
    y_test : pandas series validation labels
    param_grid : parameter grid for grid search
    min_features : minimum number of features to stop at (int)
    reduction_factor : proportion of feature to keep after each iteration
    random_state : seed for cross validation

    Returns
    -------
    Collection of feature set in a dict

    """
    # Initialize
    sk_rfe_selected_features = df.columns.tolist()
    feature_sets = {}
    
    if param_grid == None:
        param_grid = {'C': [i/100 for i in range(1,100)],
                      'gamma': ['scale'],
                      'class_weight': ['balanced'],
                      'decision_function_shape': ['ovo'],
                      'kernel': ['linear'],
                      'max_iter': [-1]}

    while len(sk_rfe_selected_features) > min_features:        
        # load selected feature from previous iteration
        X_train = X_train.loc[:,sk_rfe_selected_features]
        X_test = X_test.loc[:,sk_rfe_selected_features]
        
        # 5 fold gridsearchcv to get best model with current feature set
        svc = SVC()
        cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = random_state)
        clf = GridSearchCV(svc, param_grid = param_grid, cv = cv, scoring='balanced_accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)
        best_params = clf.best_params_

        svc = SVC(**best_params)
        
        # perform rfe with best parameter
        rfe_model = RFE(svc, n_features_to_select=int(len(sk_rfe_selected_features) * reduction_factor))
        rfe_model.fit(X_train, y_train)

        # refit model to get score
        svc = SVC(**best_params, probability = True)
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        y_score = svc.predict_proba(X_test)
        scores = [metrics.balanced_accuracy_score(y_test, y_pred),
                  metrics.roc_auc_score(y_test, y_score[:,1])]
        
        
        # Store the current set of selected features
        sk_rfe_selected_features = rfe_model.get_feature_names_out()
        feature_sets[len(sk_rfe_selected_features)] = [sk_rfe_selected_features, scores]
        fig, ax = plt.subplots(1,1, dpi = 150)        
        metrics.RocCurveDisplay.from_estimator(svc, X_test, y_test, ax = ax)
        fig.suptitle(f'Set {len(sk_rfe_selected_features)}, Balacc = {scores[0]:.3f}, AUROC = {scores[1]:.3f}')
        plt.plot()
    return feature_sets

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

df1 = pd.read_csv('part1_full_df.csv', index_col = 0)
df2 = pd.read_csv('part2_full_df.csv', index_col = 0)
meta = pd.read_csv('batch_meta.csv', index_col = 0)

temp = {'Antipsychotic responsive': 'ARE',
        'Clozapine resistant': 'CRT',
        'Clozapine responsive': 'CRE',
        'Healthy control': 'HCL'}

meta['cl'] = [temp[i] for i in meta.Class]
for cl in meta.cl.unique():
    cl_n = cl
    meta[cl_n] = [cl if meta.loc[i,'cl'] == cl else 'NOT_%s'%(cl) for i in meta.index]

df1 = missing_value_removal(df1)
df2 = missing_value_removal(df2)

shared_samples = [i for i in df1.index if i in df2.index]
shared_samples

shared_genes = [i for i in df1.columns if i in df2.columns] #dropping non shared genes
df1 = df1.loc[:,shared_genes]
df2 = df2.loc[:,shared_genes]

temp = lambda x, y: [x.iloc[i] - y.iloc[i] for i in range(len(x))]

for gene in shared_genes:
    normalisation_factor = np.median(temp(df1.loc[shared_samples,gene],
                                          df2.loc[shared_samples,gene]))
    normalise = lambda x: x + normalisation_factor
    df2.loc[:,gene] = df2.loc[:,gene].map(normalise)


df = df1.copy()
test_df = df2.copy()


test_meta = meta.loc[test_df.index]
meta = meta.loc[df.index]


meta['HCL'] = ['SCZ' if i != 'HCL' else i for i in meta.HCL]
meta['ARE'] = ['ART' if i != 'ARE' else i for i in meta.ARE]
meta['CRE'] = ['CRT' if i != 'CRE' else i for i in meta.CRE]


test_meta['HCL'] = ['SCZ' if i != 'HCL' else i for i in test_meta.HCL]
test_meta['ARE'] = ['ART' if i != 'ARE' else i for i in test_meta.ARE]
test_meta['CRE'] = ['CRT' if i != 'CRE' else i for i in test_meta.CRE]

label = 'HCL'

X_train = df # training matrix
y_train = meta.loc[:,label] # training labels
X_test = test_df # validation matrix
y_test = test_meta.loc[:,label] # validation labels

svm_rfe_feature_selection(X_train, y_train, X_test, y_test) # feature selection


# ARE/ART feature selection
label = 'ARE'

# removing healthy controls from classification
meta = meta[meta.HCL != 'HCL'] 
test_meta = test_meta[test_meta.HCL != 'HCL'] 
df = df.loc[meta.index]
test_df = test_df.loc[test_meta.index]


X_train = df # training matrix
y_train = meta.loc[:,label] # training labels
X_test = test_df # validation matrix
y_test = test_meta.loc[:,label] # validation labels

svm_rfe_feature_selection(X_train, y_train, X_test, y_test) # feature selection

# CRE/CRT feature selection
label = 'CRE'

# removing healthy controls and antipsychotic responsive labels from classification
meta = meta[(meta.HCL != 'HCL') & (meta.ARE != 'ARE')]
test_meta = test_meta[(test_meta.HCL != 'HCL') & (test_meta.HCL != 'ARE')] 
df = df.loc[meta.index]
test_df = test_df.loc[test_meta.index]


X_train = df # training matrix
y_train = meta.loc[:,label] # training labels
X_test = test_df # validation matrix
y_test = test_meta.loc[:,label] # validation labels

svm_rfe_feature_selection(X_train, y_train, X_test, y_test) # feature selection