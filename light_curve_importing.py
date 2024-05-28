#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Código para a automatização na obtenção das curvas de luz pelo telescópio espacial TESS.

"""
SEE_SLA = False
#%%
"""
Bibliotecas a serem importadas:
    * lightkurve : https://docs.lightkurve.org/whats-new-v2.html
    * numpy : https://numpy.org/doc/
    * matplotlib.pyplot :  https://matplotlib.org/stable/index.html
    * pandas : https://pandas.pydata.org/docs/
"""
import lightkurve as lk # versão 2.4.2
import numpy as np # versão 1.26.4
import matplotlib.pyplot as plt # versão 3.5.1
import pandas as pd # versão 2.2.1

#%%
df_exoplanets = pd.read_csv('/home/joshua/Documentos/Iniciação Científica/light-curves/data_exoplanets.csv')

df_aux = df_exoplanets[(df_exoplanets['Planet_Radius'] > 10) & (df_exoplanets['Planet_Radius'] < 30)]
df_exojup = df_aux.reset_index(drop=True)

planets = df_exojup.TIC
sectors = df_exojup.Sectors
period = df_exojup.Period
author = df_exojup.Detection

#%%
def light_curve(planets, author, sectors, period):
    search_result = lk.search_lightcurve(f'TIC {planets}',
                                          cadence ='short',     
                                          mission = 'TESS',     
                                          author = author, 
                                          sector = sectors
                                          )
    
    lc_collection = search_result.download_all()
    lc_original = lc_collection
    lc_normal = lc_collection.stitch() 
    
    time_initial = lc_normal.time.value[0] 
    lc_aux = lc_normal.fold(period, time_initial)
    flux = lc_aux.flux
    time = lc_aux.time
    flux_min = flux.min()
    time_at_flux_min = time.value[flux.argmin()]
    lc_superposition = lc_normal.fold(period, time_initial + time_at_flux_min)
    
    return lc_original, lc_normal, lc_superposition

#%%
def plot_light_curve(lc_set):
    for lc in lc_set:
        plt.figure()
        lc.scatter()
        plt.title('Light Curve Folder')
        plt.legend()
        plt.show()
    return plt

#%%
lc_set = []
for i in range(0,3):
    sec = [int(numero) for numero in df_exojup.Sectors[i].split(",")]
    lc_o, lc_n, lc_s = light_curve(planets[i], author[i], sec, period[i])
    lc_set.append(lc_s)

#%%
plot_light_curve(lc_set).show()

#%%
