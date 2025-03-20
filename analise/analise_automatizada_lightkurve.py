#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Código para a automatização na obtenção das curvas de luz pelo telescópio espacial TESS.
"""
#%%
"""
Variáveis de controle.
"""
ASTROLAB = True
DOWNLOAD_PLOT = False
LIMIT_Y = True

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
import math as mt # versão 3.12.4
import os # versão 3.12.4

#%%
"""
Conjunto de informações sobre exoplanetas. 
"""
if ASTROLAB :
    path = '/graduacao/joshuakipper/Documents/IC/Exoplanetas/light-curves/Dados_Exoplanetas/data_ExoFOP.csv'
else :
    path = '/home/joshua/Documentos/iniciacao_cientifica/light-curves/data_exoplanets/targets_KP.csv'
# O comando pd.read_csv() necessita de um caminho para o arquivo.arq que será aberto
df_exoplanets = pd.read_csv(path)

#%%
# Lista de dados necessários
planet_names = df_exoplanets['TOI']
planet_rays = df_exoplanets['Planet Radius (R_Earth)']
orbital_periods = df_exoplanets['Period (days)']
transits_duration = df_exoplanets['Duration (hours)']
star_names = df_exoplanets['TIC ID']
star_rays = df_exoplanets['Stellar Radius (R_Sun)']
star_masses = df_exoplanets['Stellar Mass (M_Sun)']
star_temperature = df_exoplanets['Stellar Eff Temp (K)']
star_magnitudes = df_exoplanets['TESS Mag']

#%%
def classify_star(star_temperature):

    if star_temperature >= 30000:
        return "O"
    elif 10000 <= star_temperature < 30000:
        return "B"
    elif 7500 <= star_temperature < 10000:
        return "A"
    elif 6000 <= star_temperature < 7500:
        return "F"
    elif 5200 <= star_temperature < 6000:
        return "G"
    elif 3700 <= star_temperature < 5200:
        return "K"
    elif 2400 <= star_temperature < 3700:
        return "M"
    else:
        return "ERRO"

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
        if neighbors >= 10:  # Se um ponto tiver no mínimo 10 vizinhos, terminar a busca
            best_point = point
            break 
    return best_point

#%%
def light_curve(star_name, orbital_period):
    
    # Pesquisa todas as curvas com esse conjunto de endereçoes 
    search_result = lk.search_lightcurve(f'TIC {star_name}',
                                         cadence ='short',     
                                         mission = 'TESS',     
                                         author = 'SPOC', 
                                         )
    
    # Downloado de todas as curvas encontradas na pesquisa 
    lc_collection = search_result.download_all()
    
    # Une e normaliza todas as curvas baixadas 
    lc_aux = lc_collection.stitch()
    
    # Remove os elementos nulos 
    lc_normal = lc_aux.remove_nans()
    time_initial = lc_normal.time.value[0] 
    
    # Primeira dobra para sobrepor o fluxo e reduzir o tempo
    lc_fold = lc_normal.fold(orbital_period, time_initial)
    
    # Encontrando o fluxo de menor valor no transito 
    radius = epslon(lc_fold.flux.mean(), lc_fold.flux.min())
    matrix = np.column_stack((lc_fold.time.value.tolist(), lc_fold.flux.value.tolist()))  
    points = matrix[np.argsort(matrix[:, 1])]
    
    # Tempo relativo ao menor fluxo no transito
    time_at_flux_min = neighborhood(points, radius)
    
    # Segunda dobra com centralização temporal 
    lc_superposition = lc_normal.fold(orbital_period, time_initial + time_at_flux_min[0])
    
    return lc_collection, lc_normal, lc_superposition

#%%
def plot_light_curve_superposition(lc_set, transit_duration, planet_name, orbital_period, 
                                   star_temperature, star_magnitude):
    i = -1
    j_list = [False, True]
    for lc in lc_set:
        i = i + 1
        k = -1
        for j in j_list: 
            k = k + 1
            t = lc.time.value 
            f = lc.flux
            # Constante para seccionar o transito no tempo adequeado 
            section = (transit_duration[i] / 24) * 2
            fig, axs = plt.subplots(figsize = (10,5), dpi = 200)
            exoplanet_legend = (
                f'TOI : {planet_name[i]}\n'
                f'Period : {orbital_period[i]:.2f} Days\n'
                f'Transit : {transit_duration[i]:.2f} Hours'
            ) 
            star_legend = (
                f'Star Type : {classify_star(star_temperature[i])}\n'
                #f'Star Mass : {star_mass[i]:.2f}\n'
                f'Star Mag : {star_magnitude[i]:.2f}'
            ) 
            exoplanet = axs.scatter(t, f, s = 1, label = exoplanet_legend, color = 'indigo')
            star = axs.scatter(0, 0, s = 1, label = star_legend, color = 'indigo')
            axs.set_title("Centered Superimposed Light Curve", fontsize = 16)
            first_legend = axs.legend(handles = [exoplanet], loc='upper left', fontsize = 11, edgecolor = 'black')
            axs.add_artist(first_legend) 
            axs.legend(handles = [star], loc='upper right', fontsize = 11, edgecolor = 'black')           
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
            radius = epslon(lc.flux.mean(), lc.flux.min())
            if LIMIT_Y:
                axs.set_ylim(lc.flux.min() - radius/10, 
                             abs(lc.flux.min()-lc.flux.mean()) + lc.flux.mean() + radius/10)

            if j:
                axs.set_xlim(-section, section) # Corte no eixo temporal
            # Download das curvas de luz analisadas 
            if DOWNLOAD_PLOT:
                name = [f'lc({planet_names[i]}).png',
                        f'lc_section({planet_names[i]}).png']
                path_base = '/home/joshua/Documentos/iniciacao_cientifica/light-curves/examples'
                new_directory = os.path.join(path_base, f':{planet_names[i]}')
                os.makedirs(new_directory, exist_ok=True)
                path_plots = os.path.join(new_directory, name[k])
                plt.savefig(path_plots)
                plt.close()
            plt.show()

#%%
lc_collection = []
lc_normal = []
lc_superposition = []
for i in range(len(planet_names)):
    print(i)
    lc_c, lc_n, lc_s = light_curve(star_names[i], orbital_periods[i])
    lc_collection.append(lc_c)
    lc_normal.append(lc_n)
    lc_superposition.append(lc_s)
#%%
plot_light_curve_superposition(lc_superposition, transits_duration, planet_names, orbital_periods, 
                               star_temperature, star_magnitudes)

#%%
