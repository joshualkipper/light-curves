#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Função para calcular a média dos pontos dentro de uma partição.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def fluxo_ruidoso(tempo, profundidade, duracao, periodo, sigma):
    fluxo = np.ones_like(tempo) # Fluxo base normalizado (=1)
    for i in range(len(tempo)):
        fase = (tempo[i] % periodo)
        if fase < duracao:
            fluxo[i] -= profundidade # Trânsito planetário
    ruido = np.random.normal(0, sigma, len(fluxo))
    return fluxo + ruido

#%%
def media_particao(x, y, num_pontos):
    num_particoes = 2 * int(np.trunc(np.sqrt(num_pontos)))
    particoes = np.linspace(np.min(x), np.max(x), num_particoes + 1)
    indices_particoes = np.digitize(x, particoes) 
    
    x_media_particao = []
    y_media_particao = []
    x_incerteza_particao = []
    y_incerteza_particao = []
    centro_particao = 0.5 * (particoes[1:] + particoes[:-1])
    
    for i in range(1, len(particoes)):
        
        x_particao = x[indices_particoes == i]
        y_particao = y[indices_particoes == i]
        
        if len(y_particao) > 0:
            x_media_particao.append(np.mean(x_particao))
            y_media_particao.append(np.mean(y_particao))
            x_incerteza_particao.append(np.std(x_particao))  
            y_incerteza_particao.append(np.std(y_particao)) 
        else:
            x_media_particao.append(np.nan)
            y_media_particao.append(np.nan)
            x_incerteza_particao.append(np.nan)
            y_incerteza_particao.append(np.nan)
            
    return x_media_particao, y_media_particao, x_incerteza_particao, y_incerteza_particao, particoes, centro_particao

#%%
num_pontos = 1000
x = np.linspace(0, 10, num_pontos)
y = fluxo_ruidoso(x, profundidade = 0.01, duracao=0.1, periodo=1.0, sigma=0.001)

#%%
x_media, y_media, x_incerteza, y_incerteza, particoes, centro_particao = media_particao(x, y, num_pontos)

#%%
plt.figure(figsize=(12, 6), dpi=200)
plt.scatter(x, y, color='mediumaquamarine', alpha=0.5, label='Dados Originais')
plt.errorbar(centro_particao, y_media, xerr=x_incerteza, yerr=y_incerteza, fmt='o', color='crimson', label='Média por Partição (com Incerteza)')

for i in range(len(centro_particao)):
    plt.bar(particoes[i], np.max(y) + 1, width=particoes[i + 1] - particoes[i], 
            bottom=-np.max(y) + 1, align='edge', alpha=0.2, color='blue', edgecolor='black')
    
plt.xlabel('X')
plt.ylabel('Y')
plt.ylim(np.min(y)*0.998, np.max(y)*1.002)
plt.title('Dados e Dados Médios por Partição')
plt.legend()
plt.show()
#%%
