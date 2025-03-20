#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Código para realizar a determinação do período de uma curva de luz utilizando 
a biblioteca PyAstronomy
"""
import numpy as np
import matplotlib.pylab as plt
from PyAstronomy import pyTiming as pyt
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
dt = 0.00035
tempo_max = 5.0
tempo = np.arange(0, tempo_max, dt)
profundidade = 0.01
periodo = 1.0
duracao = 0.1
sigma = 0.001
fluxo = fluxo_ruidoso(tempo, profundidade, duracao, periodo, sigma)

dp = 0.001
N_p = int(tempo_max/dp)
tps = (0.5, tempo_max, N_p) # Intervalo de Periodos Teste

#%%
p, sl = pyt.stringlength_dat(tempo, fluxo, tps)
periodo_estimado = p[np.argmin(sl)]

#%%
plt.figure(figsize=(12,6))
plt.plot(p, sl, 'b.-', label = f"Periodo Estimado = {p[np.argmin(sl)]:.2f} Dias")
plt.title("Determinação do Período via PyAstronomy")
plt.legend()
plt.ylabel("Comprimento Corda")
plt.xlabel("Periodo")
plt.show()