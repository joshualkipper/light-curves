#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gif da evolução temporal da curva de luz dobrada.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%%
# Funções definidas anteriormente
def curva_de_luz(tempo, profundidade=0.01, duracao=0.1, periodo=1.0):
    fluxo = np.ones_like(tempo)
    for i in range(len(tempo)):
        fase = (tempo[i] % periodo) / periodo
        if fase < duracao:
            fluxo[i] -= profundidade
    return fluxo

def ruido(curva_luz, sigma=0.001):
    ruido = np.random.normal(0, sigma, len(curva_luz))
    return curva_luz + ruido

def dobrar_curva_de_luz(tempo, fluxo, periodo=1.0):
    fase = (tempo % periodo)
    fase_ordem = np.argsort(fase)
    return fase[fase_ordem], fluxo[fase_ordem]

def comprimento_corda(corda):
    comprimento = 0.0
    for i in range(len(corda) - 1):
        distancia = np.linalg.norm(corda[i + 1] - corda[i])
        comprimento += distancia
    return comprimento

#%%
# Parâmetros iniciais
dt = 0.001
tempo = np.arange(0, 10, dt)
fluxo_base = curva_de_luz(tempo)
fluxo_ruidoso = ruido(fluxo_base)

#%%
# Configuração da figura
fig, ax = plt.subplots(figsize=(12, 6))
corda_total = np.column_stack((tempo, fluxo_ruidoso))
line1, = ax.plot([], [], color='blue', label=f"Comprimento Curva Total = {comprimento_corda(corda_total):.2f}")
line2, = ax.plot([], [], color='red', alpha=0.5, label="Curva Dobrada")
ax.set_xlim(0, 10)
ax.set_ylim(0.98, 1.02)
ax.set_xlabel("Tempo ou Fase")
ax.set_ylabel("Fluxo")
ax.legend()

# Função de atualização para cada quadro da animação
def update(frame):
    periodo = 10 * (frame + 1) / 40  # Variando o período de 0 até 10
    fase, fluxo_dobrado = dobrar_curva_de_luz(tempo, fluxo_ruidoso, periodo)
    
    # Calcula o comprimento da curva dobrada para o período atual
    corda = np.column_stack((fase, fluxo_dobrado))
    comprimento = comprimento_corda(corda)
    
    line1.set_data(tempo, fluxo_ruidoso)
    line2.set_data(fase, fluxo_dobrado)
    
    # Atualiza o título com o período e o comprimento da curva
    ax.set_title(f"Período = {periodo:.2f}, Comprimento da Curva Dobrada = {comprimento:.2f}")
    
    return line1, line2

#%%
# Criação da animação
anim = FuncAnimation(fig, update, frames=40, interval=100, blit=True)  # 40 quadros com intervalo de 150 ms

# Salvando a animação como GIF em um local acessível
output_path = "/graduacao/joshuakipper/Documentos/ic/exoplanetas/curva_de_luz_animacao_nova.gif"
anim.save(output_path, writer="imagemagick", fps=2)
plt.close(fig)  # Fecha a figura para evitar uma janela extra

print(f"O GIF foi salvo em: {output_path}")