#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Código para a automatização na obtenção das curvas de luz pelo telescópio espacial TESS.
"""
#%%
"""
Variáveis de controle.
"""
IN_ASTROLAB = True
#%%
"""
Bibliotecas a serem importadas:
    * lightkurve : https://docs.lightkurve.org/whats-new-v2.html
    * numpy : https://numpy.org/doc/
    * matplotlib.pyplot :  https://matplotlib.org/stable/index.html
    * pandas : https://pandas.pydata.org/docs/
    * scipy : https://docs.scipy.org/doc/scipy/
    * math : https://docs.python.org/3/library/math.html
"""
import lightkurve as lk # versão 2.4.2
import numpy as np # versão 1.26.4
import matplotlib.pyplot as plt # versão 3.5.1
import matplotlib.ticker as ticker # versão 3.5.1
import pandas as pd # versão 2.2.1
import scipy.spatial as ss # versão 1.8.0
import math as mt # versão 3.10.12

#%%
"""
Conjunto de informações sobre exoplanetas. 
"""
if IN_ASTROLAB:
    path = '/graduacao/joshuakipper/light-curves/data_exoplanets.csv'
else : 
    path = '/home/joshua/Documentos/Iniciação Científica/light-curves/data_exoplanets.csv'
# O comando pd.read_csv() necessita de um caminho para o arquivo.cvs que será aberto
df_exoplanets = pd.read_csv(path)
# Por limitaçoes escolhemos somente exoplanetas com raios maiores que 10 raios terrestres
# Para evitar absurdos limitamos o raio dos exoplanetas à 30 raios terrestres 
df_aux = df_exoplanets[(df_exoplanets['Planet_Radius'] > 10) & (df_exoplanets['Planet_Radius'] < 30) & 
                      (df_exoplanets['Period'] > 1) & (df_exoplanets['Period'] < 10)]
# Reseta o indíce para ficar contínuo de [0,n]
df_exojup = df_aux.reset_index(drop=True) 
# Lista de dados necessários
planets = df_exojup.TIC
sectors = df_exojup.Sectors
period = df_exojup.Period
author = df_exojup.Detection
time_transit = df_exojup.Duration

#%%
def epslon(flux_medium, flux_minimun):
    a = flux_medium - flux_minimun
    if a == 0:
        return 0
    # Função que tira o expoente de 'a' e pega o valor inteiro do produto com o log
    exponent = mt.floor(mt.log10(abs(a)))
    epslon = 1 * (10**exponent)
    return epslon

#%%
def neighborhood(points, radius):
    best_point = None
    for point in points:
        # Calcular a distância de 'point' para todos os outros pontos
        distances = ss.distance.cdist([point], points, 'euclidean').flatten()
        # Contar quantos pontos estão dentro do raio (excluindo o próprio ponto)
        neighbors = np.sum(distances < radius) - 1
        if neighbors > 10:  # Se um ponto tiver no mínimo 10 vizinhos, terminar a busca
            best_point = point
            break 
    return best_point

#%%
def light_curve(planets, author, sectors, period):
    
    # Pesquisa todas as curvas com esse conjunto de endereçoes 
    search_result = lk.search_lightcurve(f'TIC {planets}',
                                          cadence ='short',     
                                          mission = 'TESS',     
                                          author = 'SPOC', 
                                          sector = sectors
                                          )
    
    # Downloado de todas as curvas encontradas na pesquisa 
    lc_collection = search_result.download_all()
    
    # Une e normaliza todas as curvas baixadas 
    lc_aux = lc_collection.stitch()
    
    # Remove os elementos nulos 
    lc_normal = lc_aux.remove_nans()
    time_initial = lc_normal.time.value[0] 
    
    # Primeira dobra para sobrepor o fluxo e reduzir o tempo
    lc_fold = lc_normal.fold(period, time_initial)
    
    # Encontrando o fluxo de menor valor no transito 
    radius = epslon(lc_fold.flux.mean(), lc_fold.flux.min())
    matrix = np.column_stack((lc_fold.time.value.tolist(), lc_fold.flux.value.tolist()))  
    points = matrix[np.argsort(matrix[:, 1])]
    
    # Tempo relativo ao menor fluxo no transito
    time_at_flux_min = neighborhood(points, radius)
    
    # Segunda dobra com centralização temporal 
    lc_superposition = lc_normal.fold(period, time_initial + time_at_flux_min[0])
    
    return lc_collection, lc_normal, lc_superposition

#%%
def plot_light_curve(lc_set):
    i = -1
    for lc in lc_set:
        i = i + 1
        t = lc.time.value 
        f = lc.flux
        # Constante para seccionar o transito no tempo adequeado 
        section = (time_transit[i] / 24) * 2
        fig, axs = plt.subplots(figsize = (10,5), dpi = 200)
        text_legend = (
            f'TIC ID : {planets[i]}\n'
            f'Period : {period[i]:.2f} Days\n'
            f'Transit : {time_transit[i]:.2f} Hours'
        ) 
        axs.scatter(t, f, s = 1, label = text_legend, color = 'indigo')
        axs.set_title("Centered Superimposed Light Curve", fontsize = 16)
        axs.legend(loc='upper center', fontsize = 11, edgecolor = 'black')
        # Configurações dos eixos e da borda
        for spine in axs.spines.values():
            spine.set_color('black')   # Cor da borda
            spine.set_linewidth(1)     # Largura da borda
        axs.xaxis.set_minor_locator(ticker.AutoMinorLocator(5)) # N° de risquinhos em x
        axs.yaxis.set_minor_locator(ticker.AutoMinorLocator(5)) # || || || em y
        # Ajustando o tamanho, cor e direção dos risquinhos
        axs.tick_params(which = 'minor', length = 5, color = 'black', direction = 'in')
        axs.tick_params(which = 'major', length = 8, color = 'black', direction = 'in')
        axs.tick_params(axis = 'both', labelsize = 12)
        axs.set_xlabel("Phase[Days]", fontsize = 12)
        axs.set_ylabel("Normalized Flux", fontsize = 12)
        #axs.set_xlim(-section, section) # Corte no eixo temporal
    return plt

#%%
lc_superposition = []
for i in range(0,30):
    print(i)
    sec = [int(numero) for numero in df_exojup.Sectors[i].split(",")]
    lc_o, lc_n, lc_s = light_curve(planets[i], author[i], sec, period[i])
    lc_superposition.append(lc_s)

#%%
plot_light_curve(lc_superposition).show()

#%%
