# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:13:42 2020

@author: Techplus
"""

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

RESOURCES_PATH = 'resources/'
NOTAS_MAT = 'NU_NOTA_MT'

#%% Passo 1: Importando os dados
print('Carregando dados...')
df_train = pd.read_csv(RESOURCES_PATH + 'train.csv', sep="," , encoding="UTF8" )
df_test = pd.read_csv(RESOURCES_PATH + 'test.csv', sep="," , encoding="UTF8" )

#%% Passo 2: Correlação
"""
Correlação entre os dados considerando a nota de 
matemática (NU_NOTA_MT) como alvo
"""
print('Selecionado features de maior correlação...')
N_FEATURES = 20
corr = df_train.corr()[NOTAS_MAT]
corr = corr.sort_values(ascending=False)[0:N_FEATURES+1]

print('Exibindo correlação...')
features_corr = corr.index
fig1 = plt.figure('Correlação', figsize=(11, 8))
ax = fig1.subplots()
sns.heatmap(df_train[features_corr].corr() ,  annot=True, annot_kws={"size": N_FEATURES+1})

#%% Passo 3: Features
print('Selecionando as features...')
"""
Escolha das features com base na correlação
"""
# features = ['NU_NOTA_CN', 
#             'NU_NOTA_CH', 
#             'NU_NOTA_LC', 
#             'NU_NOTA_REDACAO', 
#             'NU_NOTA_COMP3', 
#             'NU_NOTA_COMP5',
#             'NU_NOTA_COMP4', 
#             'NU_NOTA_COMP2', 
#             'NU_NOTA_COMP1']

features = ['NU_NOTA_CN', 
            'NU_NOTA_CH', 
            'NU_NOTA_LC',
            'NU_NOTA_REDACAO', 
            'NU_NOTA_COMP3',
            'NU_NOTA_COMP5', 
            'NU_NOTA_COMP4', 
            'NU_NOTA_COMP2', 
            'NU_NOTA_COMP1',
            'TP_COR_RACA', 
            'TP_ESCOLA', 
            'IN_TREINEIRO',
            'NU_IDADE',
            'TP_LINGUA',
            'TP_SEXO',
            'TP_NACIONALIDADE',
            'TP_ST_CONCLUSAO',
            'TP_ANO_CONCLUIU',
            'IN_CEGUEIRA',
            'IN_SURDEZ',
            'IN_DISLEXIA',
            'IN_SABATISTA',
            'IN_DISCALCULIA',
            'IN_GESTANTE',
            'IN_IDOSO'
       ]

# Verificar as features estão nos dados de treino e teste
print(set(features).issubset(set(df_train.columns)))
print(set(features).issubset(set(df_test.columns)))

#%% Passo 4: Distribuição dos dados
print('Analisando distribuição dos dados...')
# Verificando notas zeros nos dados:
print('Zeros:')
print((df_train[features]==0).sum())
# Verificando valores nulos nos dados:
print('Nulos:')
print(df_train[features].isnull().sum())


# # Subistituir dados nulos por 0:
# x0 = df_train['NU_NOTA_CN'].fillna(0)
# x1 = df_test['NU_NOTA_CN'].fillna(0)

# # Subistituir dados nulos pela média:
# mn = df_train['NU_NOTA_CN'].mean()
# x0 = df_train['NU_NOTA_CN'].fillna(mn)
# mn = df_test['NU_NOTA_CN'].mean()
# x1 = df_test['NU_NOTA_CN'].fillna(mn)


# Eliminar Zeros e Nulos
# DO TREINO
df_train = df_train.loc[
      (df_train['NU_NOTA_CN'].notnull()) & (df_train['NU_NOTA_CN'] != 0) &
      (df_train['NU_NOTA_CH'].notnull()) & (df_train['NU_NOTA_CH'] != 0) &
      (df_train['NU_NOTA_LC'].notnull()) & (df_train['NU_NOTA_LC'] != 0) &
      (df_train['NU_NOTA_REDACAO'].notnull()) & (df_train['NU_NOTA_REDACAO'] != 0)
]
# DO TESTE
df_test = df_test.loc[
      (df_test['NU_NOTA_CN'].notnull()) & (df_test['NU_NOTA_CN'] != 0) &
      (df_test['NU_NOTA_CH'].notnull()) & (df_test['NU_NOTA_CH'] != 0) &
      (df_test['NU_NOTA_LC'].notnull()) & (df_test['NU_NOTA_LC'] != 0) &
      (df_test['NU_NOTA_REDACAO'].notnull()) & (df_test['NU_NOTA_REDACAO'] != 0)
]

# Trocando String por float em 'TP_SEXO'
df_train['TP_SEXO'] = df_train['TP_SEXO'].replace('F', 1005)
df_train['TP_SEXO'] = df_train['TP_SEXO'].replace('M', 4)

df_test['TP_SEXO'] = df_test['TP_SEXO'].replace('F', 1005)
df_test['TP_SEXO'] = df_test['TP_SEXO'].replace('M', 4)

# Verificando notas zeros nos dados:
print('Zeros:')
print((df_train[features]==0).sum())
# Verificando valores nulos nos dados:
print('Nulos:')
print(df_train[features].isnull().sum())

fig2 = plt.figure('Distribuição dos dados', figsize=(11, 8))
# view = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC']
view = ['NU_IDADE',
            'TP_LINGUA',
            'TP_SEXO',
            'TP_NACIONALIDADE',
            'TP_ST_CONCLUSAO',
            'TP_ANO_CONCLUIU',
            'IN_CEGUEIRA',
            'IN_SURDEZ',
            'IN_DISLEXIA',
            'IN_SABATISTA',
            'IN_DISCALCULIA',
            'IN_GESTANTE',
            'IN_IDOSO']
for i in range(len(view)):
    x0 = df_train[view[i]]
    x1 = df_test[view[i]]
    ax = plt.subplot(1,len(view), i+1)
    sns.distplot(x0, ax=ax)
    sns.distplot(x1, ax=ax)
    plt.legend(labels=['TRAIN','TEST'], ncol=2, loc='upper left')

#%% Passo 5: Modelo de Regressão
print('Gerando Modelo...')

y_train = df_train['NU_NOTA_MT']
x_train = df_train[features]
x_test = df_test[features]

# NORMALIZAÇÃO DO TREINO E TESTE
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Regressor: RandonForest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=24, n_jobs=2, max_depth=19)
# regressor = RandomForestRegressor( 
#            criterion='mae', 
#            max_depth=8,
#            max_leaf_nodes=None,
#            min_impurity_split=None,
#            min_samples_leaf=1,
#            min_samples_split=2,
#            min_weight_fraction_leaf=0.0,
#            n_estimators= 500,
#            n_jobs=-1,
#            random_state=0,
#            verbose=0,
#            warm_start=False
# )

# # Regressor: Gradient Boosting
# from sklearn.ensemble import GradientBoostingRegressor
# regressor = GradientBoostingRegressor(max_depth=20)

#%% Passo 6: Treinando Modelo
print('Treinando o modelo...')
regressor.fit(x_train, y_train)

#%% Passo 7: Predição
print('Realizando a predição do teste...')
y_pred_test = regressor.predict(x_test)

#%% Passo 8: Salvando resposta
resposta = pd.DataFrame()
resposta['NU_INSCRICAO'] = df_test['NU_INSCRICAO']
resposta['NU_NOTA_MT'] = np.around(y_pred_test, 2)
resposta.to_csv(RESOURCES_PATH + 'answer.csv', index=False, header=True)

# answer1 = pd.read_csv(RESOURCES_PATH + 'answer1.csv', sep="," , encoding="UTF8" )
# Score do Modelo
# print('MAE:', metrics.mean_absolute_error(y_train, y_pred_test).round(8)  )
# print('MSE:', metrics.mean_squared_error(y_train, y_pred_test).round(8) )  
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_test)).round(8))