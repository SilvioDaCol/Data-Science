#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[44]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

# sns.set()


# In[2]:


countries = pd.read_csv("countries.csv")


# In[3]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[4]:


# Sua análise começa aqui.
def str_to_float(x):
    return float(str(x).replace(',', '.'))

def strip(x):
    return x.strip()

columns_to_float = ['Pop_density', 'Coastline_ratio', 'Net_migration', 'Infant_mortality', 'Literacy', 
                    'Phones_per_1000', 'Arable', 'Crops', 'Other', "Climate", 'Birthrate', 'Deathrate', 'Agriculture',
                    'Industry', 'Service']
columns_to_strip = ['Country', 'Region']

countries[columns_to_float] = countries[columns_to_float].applymap(str_to_float)
countries[columns_to_strip] = countries[columns_to_strip].applymap(strip)


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[5]:


def q1():
    # Retorne aqui o resultado da questão 1.
    regions = countries.Region.sort_values()
    return list(regions.unique())


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[122]:


def q2():
    # Retorne aqui o resultado da questão 2.
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    discretizer.fit(countries[['Pop_density']])
    ds_pop_density = discretizer.transform(countries[["Pop_density"]])[:, 0]
    edges = discretizer.bin_edges_[0]
    return int(sum(ds_pop_density == 9))


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[126]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return len(countries.Region.unique()) + len(countries.Climate.unique())


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[8]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[9]:


def q4():
    # Retorne aqui o resultado da questão 4.
    pipeline = Pipeline(steps=[
        ("median_imputer", SimpleImputer(strategy="median")),
        ("standardizer", StandardScaler())
    ])

    int_float_df = countries.select_dtypes(include=['int64', 'float64'])
    pipeline.fit_transform(int_float_df)

    test_processed = pipeline.transform([test_country[2:]])

    arable_index = int_float_df.columns.to_list().index('Arable')

    return float(np.round(test_processed[0][arable_index], 3))


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[129]:


def q5():
    # Retorne aqui o resultado da questão 4.
    qtl1 = countries.Net_migration.quantile(0.25)
    qtl3 = countries.Net_migration.quantile(0.75)
    iqr = qtl3 - qtl1
    low_edge = qtl1 - 1.5 * iqr
    high_edge = qtl3 + 1.5 * iqr

    outliers_abaixo = len(countries.Net_migration[(countries.Net_migration < low_edge)])
    outliers_acima = len(countries.Net_migration[(countries.Net_migration > high_edge)])
    return (outliers_abaixo, outliers_acima, False)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[123]:


def q6():
    # Retorne aqui o resultado da questão 4.
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    count_vectorizer = CountVectorizer()
    newsgroup_count = count_vectorizer.fit_transform(newsgroup.data)

    phone_index = count_vectorizer.vocabulary_.get('phone')
    phone_column = newsgroup_count[:, phone_index].toarray()
    return int(phone_column.sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[130]:


def q7():
    # Retorne aqui o resultado da questão 4.
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(newsgroup.data)
    newsgroups_vectorized = vectorizer.transform(newsgroup.data)
    
    phone_index = vectorizer.vocabulary_.get('phone')
    
    return float(np.round(newsgroups_vectorized[:, phone_index].toarray().sum(), 3))


# In[ ]:




