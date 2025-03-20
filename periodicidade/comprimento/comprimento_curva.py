#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Função para calcular a distância entre pontos de uma corda
"""
#%%
import numpy as np

#%%
def calcula_comprimento_corda(corda):
    comprimento = 0.0
    # Percorre cada par de pontos consecutivos
    for i in range(len(corda) - 1):
        # Calcula a distância euclidiana entre dois pontos consecutivos (x_i, y_i) e (x_{i+1}, y_{i+1})
        distancia = np.linalg.norm(corda[i + 1] - corda[i])
        comprimento += distancia
    return comprimento

#%%
# Exemplo de pontos representando uma corda linear
corda_linear = np.array([
    [0, 0],    # Ponto inicial (x1, y1)
    [1, 1],    # Ponto final (x2, y2)
])

# Calcula o comprimento da corda
comprimento_total_linear = calcula_comprimento_corda(corda_linear)
print("Comprimento total da corda:", comprimento_total_linear)

#%%
# Exemplo de pontos representando uma corda senoidal
A = 1.0        # Amplitude
omega = 2 * np.pi / 10  # Frequência angular (um ciclo a cada 10 unidades de tempo)
phi = 0.0      # Fase inicial
t_inicio = 0   # Tempo inicial
t_fim = 10     # Tempo final
num_pontos = 100  # Número de pontos para amostrar a função

t = np.linspace(t_inicio, t_fim, num_pontos)  # Valores de tempo
x = t                                         # Assumimos x como o tempo
y = A * np.sin(omega * t + phi)               # Calcula y como uma função senoidal de t
corda_senoidal = np.column_stack((x, y))               # Cria uma lista de pontos (x, y)

# Calcula o comprimento da corda
comprimento_total_senoidal = calcula_comprimento_corda(corda_senoidal)
print("Comprimento total da corda senoidal:", comprimento_total_senoidal)

#%%
