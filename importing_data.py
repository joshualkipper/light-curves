#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Código para a importação das curvas de luz de um modo automatizado.

"""
#%%
SEE_INFORMATION = False

#%%
import pandas as pd

#%%
"""
Modo de importar os dados dos exoplanetas do site ExoFop. A tabela com os dados deve ser
baixado localmente e deve ser informado o caminho(pwd) para a função.

"""
df_exoplanets = pd.read_csv('/home/joshua/Documentos/Iniciação Científica/light-curves/data_exoplanets.csv')

#%%
planets = df_exoplanets.TIC
sectors = df_exoplanets.Sectors
period = df_exoplanets.Period
author = df_exoplanets.Detection

#%%
if SEE_INFORMATION:
    x = 1
    information_exoplanet = df_exoplanets.iloc[x]
    print(information_exoplanet)

#%%
"""
Os setores importados direto do dataframe são tratados como string, essa é uma maneira
de transformar os strings em uma lista para que os dados possas ser utilizados.

"""
y = 1
sec = [int(term) for term in sectors[y].split(",")]
print(sec)

#%%
"""

"""
df_aux = df_exoplanets[(df_exoplanets['Planet_Radius'] > 10) & (df_exoplanets['Planet_Radius'] < 30)]
df_exojup = df_aux.reset_index(drop=True)

#%%