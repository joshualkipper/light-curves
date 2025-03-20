#!/usr/bin/env python3
# -*- coding: utf-8
"""
Código para determinar o período de trânsitos planetários utilizando o método
de maximização do comprimento da corda formado por pontos representativos.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
SINAL_UNICO = True
if SINAL_UNICO :
    SINAL = 0 # Degrau = 0 | Parábolico = 1 | Vazio = 2
    PLOT_TRIPLO_MAXIMO = True
    PLOT_LOCALIZADO = True
    PLOT_MAXIMO = True
else:
    TODOS_SINAIS = True

#%%
"""
Função responsável por simular o fluxo de uma curva de luz com ruído gaussiano. 
Fazemos isso utilizando o tempo de duração e o período dos trânsitos, assim 
calculamos a sua localização temporal (fase) e caso esteja ocorrendo um trânsito
descontamos um δf (profundidade) do fluxo médio, que está normalizado (=1). Ao
final adicionamos um vetor de ruídos gaussianos ao vetor de fluxos. A função 
permite 3 tipos de trânsitos (profundidades):
    (i) Trânsito por uma função degrau;
    (ii) Trânsito por uma função parabólica;
    (iii) Ausência de trânsito.
"""
def fluxo_ruidoso(tempo, profundidade, duracao, periodo, sigma, transito):
    fluxo = np.ones_like(tempo) # Fluxo base normalizado (=1)
    A = np.array([[1,0,0],[1, duracao/2, (duracao/2)**2], [1, duracao, duracao**2]])
    b = np.array([0, profundidade, 0])
    coeficientes = np.linalg.solve(A, b)
    for i in range(len(tempo)):
        fase = (tempo[i] % periodo)
        if fase < duracao:
            if transito == 'degrau': 
                fluxo[i] -= profundidade # Trânsito planetário
            elif transito == 'parabola':
                fluxo[i] -= coeficientes[0] + coeficientes[1]*fase + coeficientes[2]*fase**2
            elif transito == 'vazio':
                fluxo[i] = fluxo[i]
    ruido = np.random.normal(0, sigma, len(fluxo))
    return fluxo + ruido

#%%
"""
Função responsável por realizar a superposição dos pontos da curva de luz (CL) em uma
única curva no intervalo [0, periodo].
"""
def dobrar_CL(tempo, fluxo, periodo): 
    fase = (tempo % periodo)  # Calcula a fase de cada ponto
    fase_ordem = np.argsort(fase)  # Ordena as fases em ordem crescente
    return fase[fase_ordem], fluxo[fase_ordem]

#%%
"""
Função para calcular o comprimento de uma curva de luz (CL). Fazemos isso calculando
a distancia (euclidiana em um espaço bidimensional) entre dois pontos consecutivos
varrendo sobre todos os pontos da CL. Ao final somamos todos os comprimentos para 
obter o comprimento total da CL. 
"""
def comprimento_CL(tempo, fluxo):
    curva = np.column_stack((tempo, fluxo)) # Malha bidimensional da CL
    variacao = np.diff(curva, axis=0) # Diferença entre componentes dos pontos consecutivos
    distancias = np.linalg.norm(variacao, axis=1) # Norma entre pontos consecutivos
    return np.sum(distancias)

#%%
"""
Função para cálcular os pontos representativos de uma curva de luz (CL) dentro 
de uma determinada partição temporal. Realizamos o particionamento do eixo temporal
em M partições, após isso realizamos a média de todos os pontos dentro de uma 
partição fazendo isso para todas as partições, assim trocamos N pontos da CL por
M<<N pontos representativos da CL por partição temporal.
"""
def CL_representativa(tempo, fluxo):
    num_pontos = len(tempo)
    num_particoes = int(np.trunc(np.sqrt(num_pontos)))
    particoes = np.linspace(np.min(tempo), np.max(tempo), num_particoes + 1)
    # Retorna, para cada ponto de tempo, o índice da partição à qual ele pertence
    indices_particoes = np.digitize(tempo, particoes) 
    tempo_media = []
    fluxo_media = []
    for i in range(1, len(particoes)):
        # Seleciona os valores de tempo e fluxo que pertencem à partição atual
        tempo_particao = tempo[indices_particoes == i]
        fluxo_particao = fluxo[indices_particoes == i] 
        tempo_media_particao = np.mean(tempo_particao)
        fluxo_media_particao = np.mean(fluxo_particao)
        tempo_media.append(tempo_media_particao)
        fluxo_media.append(fluxo_media_particao)
    return np.array(tempo_media), np.array(fluxo_media)

#%%
"""
Função responsável por encotrar o período, se ele existir, de uma curva de luz (CL).
Fazemos isso realizando a dobragem da CL por um período teste, fazemos esse período
variar de um período minímo até um tempo total da CL, após isso realizamos a 
substituição da CL por seus pontos representativos por partição temporal. O período
responsável por maximizar o comprimento da CL será o período real.
"""
def maximizar_comprimento_CL(tempo, fluxo, periodo_min, periodo_max, dp):
    periodos = np.arange(periodo_min, periodo_max, dp)
    comprimentos = []

    for periodo in periodos:
        fase, fluxo_dobrado = dobrar_CL(tempo, fluxo, periodo)
        fase_representativa, fluxo_representativo = CL_representativa(fase, fluxo_dobrado)
        # Normalizamos os comprimento da CL para observar os detalhes
        comprimentos.append(comprimento_CL(fase_representativa, fluxo_representativo) / periodo)
        
    indice_maior = np.argmax(comprimentos) # Indíce associado ao maior comprimento
    return periodos[indice_maior], periodos, comprimentos[indice_maior], comprimentos

#%%
"""
Parâmetos do trânsito planetário.
"""
dt = 0.00035 # Cadência em dias (≈ 30 segundos)
tempo_max = 5.0 # Tempo de exposição da curva de luz
tempo = np.arange(0, tempo_max, dt)  # Tempo de observação da curva de luz 
profundidade = 0.01 # Profundidade do trânsito 
periodo = 1.0 # Em dias
duracao = 0.1 # Duração do trânsito em dias (= 2 horas 24 minutos)
sigma = 0.001 # Nível de ruído no fluxo
transito = ['degrau', 'parabola', 'vazio']

"""
Parâmetros da simulação.
"""
periodo_min = 0.5 # Período mínimo utilizado nos testes
periodo_max = tempo_max # Período maxímo utilizado nos testes
dp = 0.001 # Sempre ≤ que a cadência
#%%
"""
Integração do algoritmo.
"""
if SINAL_UNICO:
    fluxo = fluxo_ruidoso(tempo, profundidade, duracao, periodo, sigma, transito[SINAL]) 
    fase, fluxo_dobrado = dobrar_CL(tempo, fluxo, periodo)
    periodo_real_Mx, periodos_Mx, menor_comprimento_Mx, comprimentos_Mx = maximizar_comprimento_CL(tempo, fluxo, periodo_min, periodo_max, dp)

else:
    eixo_periodos = []
    periodos_determinados = []
    eixo_comprimentos = []
    menores_comprimentos = []
    for sinal in transito:
        fluxo = fluxo_ruidoso(tempo, profundidade, duracao, periodo, sigma, sinal) 
        fase, fluxo_dobrado = dobrar_CL(tempo, fluxo, periodo)
        periodo_real_Mx, periodos_Mx, menor_comprimento_Mx, comprimentos_Mx = maximizar_comprimento_CL(tempo, fluxo, periodo_min, periodo_max, dp)
        eixo_periodos.append(periodos_Mx)
        eixo_comprimentos.append(comprimentos_Mx)
        periodos_determinados.append(periodo_real_Mx)
        menores_comprimentos.append(menor_comprimento_Mx)
#%%
"""
Curva de luz original.
"""
#fase, fluxo_dobrado = dobrar_CL(tempo, fluxo, periodo)
#%%
"""
Determinar o período minimizando o comprimento.
"""
#periodo_real_Mn, periodos_Mn, menor_comprimento_Mn, comprimentos_Mn = minimizar_comprimento_CL(tempo, fluxo, periodo_min, periodo_max, dp)

# Definir a janela ao redor do período estimado para o gráfico
janela = 0.8  # Define a largura da janela em torno do período encontrado
indice_inicio = np.searchsorted(periodos_Mx, periodo_real_Mx - janela)
indice_fim = np.searchsorted(periodos_Mx, periodo_real_Mx + janela)

#%%
"""
Determinar o período maximizando o comprimento.
"""
#periodo_real_Mx, periodos_Mx, menor_comprimento_Mx, comprimentos_Mx = maximizar_comprimento_CL(tempo, fluxo, periodo_min, periodo_max, dp)

#%%
"""
Função para plotar a curva de luz original e dobrada.
"""
def plot_CL(tempo, fluxo, fase, fluxo_dobrado, periodo, titulo="Curva de Luz"):
    plt.figure(figsize=(12, 12), dpi=200)
    
    # Curva de luz original com ruído
    plt.subplot(3, 1, 1)
    plt.plot(tempo, fluxo, label='Curva de Luz', color='lightsteelblue')
    plt.xlabel('Tempo')
    plt.ylabel('Fluxo')
    plt.legend()
    
    # Curva de luz dobrada com o período real
    plt.subplot(3, 1, 2)
    plt.scatter(fase, fluxo_dobrado, s=5.0, 
                label=f'Curva de Luz Dobrada (Período estimado = {periodo:.4f})', color='royalblue')
    plt.xlabel('Fase')
    plt.ylabel('Fluxo')
    plt.legend()
    plt.suptitle(titulo)

#%%  
"""
Plota o comprimento da curva em função do período.
"""
def plot_comprimento_periodo_CL(periodos, comprimentos, periodo_real, comprimento_real, log_scale=False, label="Comprimento da Curva"):
    plt.plot(periodos, comprimentos, label=label, color='blueviolet')
    plt.axvline(x=periodo_real, color='r', alpha = 0.5, linestyle='--', label=f'Período Estimado = {periodo_real:.2f}')
    if comprimento_real is not None:
        plt.axhline(y=comprimento_real, color='b', linestyle='--', label=f'Comprimento da Curva de Luz = {comprimento_real:.4f}')
    if log_scale:
        plt.yscale('log')
    plt.xlabel('Período')
    plt.ylabel('Comprimento da Curva')
    plt.legend()
    
#%%
"""
Plota o comprimento em função do período para 3 sinais distintos.
"""
def plot_todos_sinais(eixo_periodos, eixo_comprimentos):
    plt.plot(eixo_periodos[0], eixo_comprimentos[0], label=f'Degrau | Periodo = {periodos_determinados[0]:.2f} dias', color='indigo', linestyle='-', linewidth=2, alpha=0.8)
    plt.plot(eixo_periodos[1], eixo_comprimentos[1], label=f'Parábola | Periodo = {periodos_determinados[1]:.2f} dias', color='orangered', linestyle='-', linewidth=2, alpha=0.5)
    plt.plot(eixo_periodos[2], eixo_comprimentos[2], label='Vazio', color='green', linestyle='-', linewidth=2, alpha=0.9)
    plt.legend(title="Tipos de sinais", fontsize=12, title_fontsize=14, loc='upper right', frameon=True)
    plt.title('Diferentes Tipos de Sinais no Método de Maximização de Comprimento', fontsize=14, fontweight='bold')
    plt.xlabel('Período', fontsize=12)
    plt.ylabel('Comprimento', fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
#%%
if SINAL_UNICO:
    if PLOT_TRIPLO_MAXIMO:
        plot_CL(tempo, fluxo, fase, fluxo_dobrado, periodo, titulo="Método de Maximização do Comprimento")
        
        plt.subplot(3, 1, 3)
        if PLOT_LOCALIZADO:
            plot_comprimento_periodo_CL(periodos_Mx[indice_inicio:indice_fim], comprimentos_Mx[indice_inicio:indice_fim], periodo_real_Mx, None, log_scale=True, label="Comprimento / Período")
        else:
            plot_comprimento_periodo_CL(periodos_Mx, comprimentos_Mx, periodo_real_Mx, None, log_scale=True, label="Comprimento / Período")
        
        plt.tight_layout()
        plt.show()
    
    if PLOT_MAXIMO:
        plt.figure(figsize=(12, 6), dpi=200)
        plot_comprimento_periodo_CL(periodos_Mx, comprimentos_Mx, periodo_real_Mx, None, log_scale=True, label="Comprimento / Período")
        plt.show()
    
if TODOS_SINAIS:
    plt.figure(figsize=(12, 6), dpi=200)
    plot_todos_sinais(eixo_periodos, eixo_comprimentos)
    plt.tight_layout()
    plt.show()

#%%