#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:19:26 2019

@author: meng
"""
import pandas as pd
import numpy as np


import warnings
from missingpy import MissForest
from linearmodels import PanelOLS

def panel_data(train, years_ahead=1):
    """
    It uses a random forest trained on the observed values of a data matrix (selected series codes except those
    in submit_rows_index) to predict the missing values.
    after that, use panel data model for prediction 
    Returns:
      y_pred: prediction values of target
    """
    train_melt=pd.melt(train.iloc[:,0:38],id_vars=['Country Name','Series Code'], 
        value_vars=train.columns[0:36],
       var_name='year', value_name='value')
    train_melt['year']=train_melt['year'].str[:4].astype(int)
    panel=train_melt.groupby(['Country Name','year','Series Code'])['value'].mean().unstack()
    
    # only use code with at least one observed value across 36 years in each country for the imputation data matrix
    left_feature=panel.iloc[:,9:].isna().groupby('Country Name').sum().max(axis=0)<=18
    pred=panel.iloc[:,9:].iloc[:,left_feature.values]
    
    # construct matrix of features across countries
    df=[]
    ct_list=list(set(pred.index.get_level_values(0)))
    ct_list=sorted(ct_list)
    for i in ct_list:
        df.append(pred.loc[i]) 
    predictors=pd.concat(df,axis=1)
    
    # random forest imputation 
    imputer = MissForest()
    predictors_imputed=imputer.fit_transform(predictors)
    
    panel.reset_index(inplace=True) 
    panel.columns=['Country Name','year']+['y'+str(i) for i in range(1,10)]+['x'+str(i) for i in range(1,1297)]
    nfeature=int(predictors.shape[1]/214)
    split=list(range(nfeature,predictors_imputed.shape[1],nfeature))
    _=np.split(predictors_imputed,split,1)
    predictors_new=pd.DataFrame(np.vstack(_))
    predictors_new['year']=panel.year
    predictors_new['Country Name']=panel['Country Name']
    predictors_new.columns=['x'+str(i) for i in range(1,pred.shape[1]+1)]+['year','Country Name']
    
    # combine the updated feature matrix and responses
    feature=predictors_new.isna().sum()<=0    # change to 1
    panel_left=predictors_new.iloc[:,feature.values]
    panel_comb=pd.merge(panel.iloc[:,0:11], panel_left.shift(years_ahead))
    
    # Split prediction and target
    panel_train=panel_comb.loc[panel_comb.year<2007]
    panel_train=panel_train.set_index(['Country Name','year'])
    panel_test=panel_comb.loc[panel_comb.year==2007]
    panel_test=panel_test.set_index(['Country Name','year'])

    # panel data model 
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        Ypred=pd.DataFrame()
        for i in range(1,10):
            formula='y'+str(i)+'~1+'+'+'.join(panel_train.columns[11:].values)+'+EntityEffects'
            mod = PanelOLS.from_formula(formula,panel_train)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            Ypred['y'+str(i)]=res.predict(data=panel_test).predictions
        
    # Eval
    Yval=panel_test.iloc[:,:9]
    rmse = np.sqrt(np.nanmean(np.power(Ypred - Yval, 2)))
    print(rmse)
    
    return Ypred
