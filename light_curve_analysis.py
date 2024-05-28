#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Código para obtenção de curvas de luz pela bibilioteca lightkurve. Eventualmente esse
código tem como função realizar uma analise mais refinada em alguma curva de luz 
específica que por ventura apresente problemas no método geral de importação de curvas 
de luz. Nesse código os métodos são mais manuais.

"""
#%%
"""
Bibliotecas a serem importadas:
    * lightkurve : https://docs.lightkurve.org/whats-new-v2.html
    * numpy : https://numpy.org/doc/
    * matplotlib.pyplot :  https://matplotlib.org/stable/index.html
    * pandas : https://pandas.pydata.org/docs/
    * scipy :

"""
import lightkurve as lk # versão 2.4.2
import numpy as np # versão 1.26.4
import matplotlib.pyplot as plt # versão 3.5.1
import pandas as pd # versão 2.2.1
from scipy.spatial import distance # versão 1.8.0

#%%
"""
Variáveis de controle.

"""
SHOW_LC_PLOT = True
SHOW_LC_INFORMATION = False
ALTERNATIVE_METHOD = False

#%%

"""
Função utilizada para pesquisar as curvas de luz disponíveis 
    * lk.search_lightcurve(target, radius=None, exptime=None, 
                            cadence=None, mission=('Kepler', 'K2', 'TESS'), 
                            author=None, quarter=None, month=None, campaign=None, 
                            sector=None, limit=None)[source]

"""
search_result = lk.search_lightcurve('TIC 38846515',    # ID do alvo
                                      cadence ='short',     # ‘long’|‘short’|‘fast'|float
                                      mission = 'TESS',     # Missão autora dos dados
                                      author = 'SPOC',      # Cada autor usa uma grandeza de fluxo  
                                      sector = (1,2,3,4,5,6,7,8,9,10)    # quarter|sector|campaign
                                      )

if SHOW_LC_INFORMATION:
    print(search_result) # Mostra a tabela com as informações da curva de luz 

#%%
"""
Para fazer download de todas as curvas de luz, utiliza-se o comando : 
    *.download_all() (vai ao final do nome do arquivo)

"""
lc_collection = search_result.download_all() #.stitch()

if SHOW_LC_INFORMATION:
   print(lc_collection)
   print(f"Número de curvas de luz: {len(lc_collection)}")

#%%
"""
Cada Curva de Luz tem a seguinte estrutura do tipo "tabela" 

"""
if SHOW_LC_INFORMATION:
    print(lc_collection[0]) # O algarismo vai de 0 até N-1 onde N é o número de curvas de luz, ou seja, são os índices

#%%
if SHOW_LC_INFORMATION:
    print(lc_collection[0].columns) # Exibe o número de todas as colunas da curva de luz de índice N

#%%
"""
Para visualizar o gráfico com as curvas de luz utiliza-se o comando :
    .plot() (vai ao final do nome do arquivo)
Embora parece diferente de usar um plot comum, é a mesma coisa (no quesito comandos),
há algumas diferenças a serem investigadas
    
"""
if SHOW_LC_PLOT: 
    lc_collection.plot() #Pode-se escolher a curva de luz N com [N], após o nome do objeto

#%%
"""
Para combinar todas as curvas de luz em uma única curva de luz, utiliza-se o comando :
    .stitch() 
o qual retorna um objeto da classe lightkurve.lightcurve.TessLightCurve. O comando sem argumentos irá retornar uma 
curva normalizada de todas da união de todas as curvas.

"""
if ALTERNATIVE_METHOD:
    lc = lc_collection.stitch(corrector_func=None)

lc_normal = lc_collection.stitch() 

#%%
if SHOW_LC_PLOT:
    lc_normal.scatter() # Plot com curvas de luz normalizadas 

#%%
"""
Outra maneira de normalizar a curva pela função 
    .normalize()
porém o stich parece melhor.
"""
if ALTERNATIVE_METHOD:
    lc_normalize = lc_collection[0].normalize() 

#%%
"""
Método para trabalho com as colunas do objeto 'lightkurve.lightcurve.TessLightCurve' em 
forma de array.

"""
if ALTERNATIVE_METHOD:
    period = 38.478761309913
    df_aux = lc_normal.to_pandas()
    time = (df_aux.index % period)
    flux = df_aux.flux.to_numpy()
    df_LC = pd.DataFrame({'time': time , 'flux': flux})
    df_LC = df_LC.sort_values(by='time')
    #print(df_LC)
    
    if SHOW_LC_PLOT:
        plt.figure(figsize=(10,4))
        plt.scatter(df_LC['time'] , df_LC['flux'], s=1, label = "TIC 27769688", color = 'black')
        plt.legend(fontsize = 10)
        plt.title('Light Curve Superposition')
        plt.xlabel('Time(Days)')
        plt.ylabel('Normalized Flux')
        #plt.xlim(1.6, 1.95)
        plt.show()

#%%
period = 2.8493825 # perído orbital do exoplaneta em dias
time_inicial = lc_normal.time.value[0] # tempo incial da curva de luz
lc_aux = lc_normal.fold(period, time_inicial) # em geral esses dados estão descentralizados
if SHOW_LC_PLOT:
    plt.figure()
    lc_aux.scatter()
    plt.legend()
    plt.title("Off-Center Light Curve Superposition")
    plt.show()

#%%
"""
Método genérico para centralizar a curva de luz no eixo temporal. Estamos pegando a 
coordenada do menor valor de fluxo encontrodado no objeto e jundo disso, o valor da 
coordenada temporal, ou seja, uma tupla 
    (tempo associado ao fluxo minímo, fluxo minímo)
após isso faremos com que esse valor de tempo específico seja nulo, ou seja, transladamos
o minímo do fluxo para a origem.

"""
flux_aux = lc_aux.flux
time_aux = lc_aux.time
time_min = time_aux.value[flux_aux.argmin()]

time_transit = 3.567 # em horas
section = (time_transit/24) * 5 # o 2 foi escolhido manualmente
lc_superpostion = lc_normal.fold(period, time_inicial + time_min) # dados devidamente centralizados
if SHOW_LC_PLOT:
    plt.figure()
    lc_superpostion.scatter()
    plt.legend()
    plt.title("Centered Superimposed Light Curve")
    #plt.xlim(-section, section)
    plt.show()

#%%
"""
Função qua irá analisar a vizinhança 
    ε > |p - p_o| tq ε > 0
onde p e p_o pertencem a matriz 'points'. A funça tem que ser otimizada, pois como fuga
para não analizar todos os pontos de 'points' foi estabelecido que se há 30 vizinhos 
de p_o, então esse será o ponto escolhido.

"""
def neighborhood(points, radius):
    max_neighbors = 0
    best_point = None
    
    for point in points:
        # Calcular a distância de 'point' para todos os outros pontos
        distances = distance.cdist([point], points, 'euclidean').flatten()
        
        # Contar quantos pontos estão dentro do raio (excluindo o próprio ponto)
        neighbors = np.sum(distances < radius) - 1
        
        if neighbors == 30:  # Se um ponto tiver 30 vizinhos, terminar a busca
            best_point = point
            max_neighbors = neighbors
            break
        
        if neighbors > max_neighbors:
            max_neighbors = neighbors
            best_point = point
            
    return best_point, max_neighbors

#%%

f = lc_aux.remove_nans() # remover nans para não atrapalhar na busca

time_column = f.time.value.tolist()
flux_column = f.flux.value.tolist()
matrix = np.column_stack((time_column, flux_column))

new_index = np.argsort(matrix[:, 1])
points = matrix[new_index] # um matriz Nx2, com os pontos de interesse

#%%
radius = 1.0e-4 # raio escolhido de maneira arbitrária
best_point, max_neighbors = neighborhood(points, radius)
new_time_min = best_point[0]

#%%
"""
Segunda tentativa de centralizar os dados.

"""
lc_superpostion = lc_normal.fold(period, time_inicial + new_time_min) # dados devidamente centralizados
if SHOW_LC_PLOT:
    plt.figure()
    lc_superpostion.scatter()
    plt.legend()
    plt.title("Centered Superimposed Light Curve")
    #plt.xlim(-section, section)
    plt.show()
    
#%%