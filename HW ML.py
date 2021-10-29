# -*- coding: utf-8 -*-

"""
Created on Wed Sep 22 14:21:25 2021

@author: elfab
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import plotly.express as px
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

import scipy.stats as stats


##### Préparation des données

df = pd.read_csv('world-happiness-report-2021.csv')
#df.head()
#df.info()
#df.describe()

df2 = pd.read_csv('world-happiness-report.csv')
df2.head()

#Création liste des régions du monde par pays
country_region = df.iloc[:,0:2]

#Mise en forme des 2 datasets en prévision du concat
#dataset de 2021 :
df_1 = df.iloc[:,:12]
df_1 = df_1.drop(df.iloc[:,3:6], axis=1)
df_1 = df_1.drop('Regional indicator', axis=1)
df_1['year']='2021'

#dataset historique depuis 2005 :
df_2 = df2.iloc[:, :-2]
dico = {'Life Ladder' : 'Ladder score', 'Log GDP per capita' : 'Logged GDP per capita', 'Healthy life expectancy at birth' : 'Healthy life expectancy'}
df_2.rename(dico, axis=1, inplace=True)

#création du fichier contenants les données depuis 2005 :
df_full = pd.concat([df_1, df_2], axis=0)
df_full['year'] = df_full['year'].astype('int')
df_full = df_full.sort_values(by='Country name', ascending=True)

#ajout colonne Region d'après le df country_region créé précédemment :
df_full = df_full.merge(country_region, on = 'Country name', how ='inner')

#ajout du rank par année de chaque pays
df_full["rank"] = df_full.groupby("year")["Ladder score"].rank("dense", ascending=False).astype('int')

#tri de df_full par année et par nom de pays
df_full = df_full.sort_values(['year', 'Country name'])

#Remplacement des NaN par les moyennes des variables en fonction de chaque pays
for i in df_full.iloc[:,2:8].columns : 
    df_full[i] = df_full[i].fillna(df_full.groupby('Country name')[i].transform("mean"))
    
df_full.head()
df_full.info()

##### Modèles de ML

## définition des variables avec le Ladder score comme cible

col_feat = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']

target_score = df_full['Ladder score']
features = df_full[col_feat]

X_train, X_test, y_train, y_test = train_test_split(features, target_score, test_size=0.2, shuffle=False)

scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

### modèle de ML

## Régression linéaire

model = LinearRegression()

model.fit(X_train_scaled, y_train)

print('Coef de détermination du modèle :', model, model.score(X_train_scaled, y_train))
print('Coef de détermination obtenu par cv :', model, cross_val_score(model, X_train_scaled, y_train).mean())
print('Score test :', model, model.score(X_test_scaled, y_test))

## SVM

model = SVR(C=1, gamma = 0.5, kernel = 'rbf')

model.fit(X_train_scaled, y_train)

print('Coef de détermination du modèle :', model, model.score(X_train_scaled, y_train))
print('Coef de détermination obtenu par cv :', model, cross_val_score(model, X_train_scaled, y_train).mean())
print('Score test :', model, model.score(X_test_scaled, y_test))


# définition meilleurs paramètres 
#parametres = {'C':[1,100,5], 'kernel':['rbf', 'linear'], 'gamma' : [0.001,0.1,0.5,10,100]}
#grid_clf = model_selection.GridSearchCV(estimator=model, param_grid=parametres)
#grille = grid_clf.fit(X_train_scaled, y_train)
#print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,['params', 'mean_test_score']]) 
#print('Meilleurs paramètres SVM :',grid_clf.best_params_)


print('modèle actif :', model)
pred_train = model.predict(X_train_scaled)
pred_test = model.predict(X_test_scaled)

df_score_pred = pd.DataFrame({'Ladder score' : y_test, 'Score prédit' : pred_test}, index = y_test.index)

df_score_pred['diff'] = df_score_pred['Score prédit'] - df_score_pred['Ladder score']

df_score_pred.head(10)

plt.plot(df_score_pred['Ladder score']);



##### fonction prédiction Happy score

def score(model, GDP, social, life, choice, generosity, corruption) :
    x = np.array([GDP, social, life, choice, generosity, corruption]).reshape(1,6)
    x = scaler.transform(x)
    return model.predict(x)

GDP = 10.7
social = 0.962
life = 75
choice = 0.957
generosity = 0.256
corruption = 0.503

Happy_score = score(model, GDP, social, life, choice, generosity, corruption)

df_score_pred