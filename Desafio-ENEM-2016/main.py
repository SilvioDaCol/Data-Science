#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


# 5 primeiros e 5 últimos registros...
pd.concat([black_friday.head(), black_friday.tail()])


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[4]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[5]:


# Verificar quais as faixas de idade existentes:
black_friday['Age'].unique()


# In[6]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return black_friday[((black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35'))].shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[7]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[8]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[12]:


def q5():
    # Retorne aqui o resultado da questão 5.
    na_values = black_friday.isna().any(axis=1).value_counts()[True]
    return float(na_values/black_friday.shape[0])


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[34]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return int(black_friday.isna().sum().max())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[35]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return black_friday['Product_Category_3'].value_counts().index[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[36]:


def q8():
    # Retorne aqui o resultado da questão 8.
    purchases = black_friday['Purchase']
    norm_purchases = (purchases - purchases.min()) / (purchases.max() - purchases.min())
    return float(norm_purchases.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[37]:


def q9():
    # Retorne aqui o resultado da questão 9.
    purchases = black_friday['Purchase']
    stded_purchases = (purchases - purchases.mean()) / purchases.std()
    return int(stded_purchases[(stded_purchases <= 1) & (stded_purchases >= -1)].count())


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[38]:


def q10():
    # Retorne aqui o resultado da questão 10.
    cat_2_3 = black_friday[['Product_Category_2', 'Product_Category_3']]
    cat_2_na2 = cat_2_3[cat_2_3['Product_Category_2'].isna()]['Product_Category_2']
    cat_3_na2 = cat_2_3[cat_2_3['Product_Category_2'].isna()]['Product_Category_3']

    return cat_2_na2.equals(cat_3_na2)


# In[ ]:




