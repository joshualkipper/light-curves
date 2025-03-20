#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Código para obtenção de curvas de luz pela bibilioteca lightkurve.

"""
#%%
"""
Problema : Por algum motivo algumas curvas de luz estão vindo já normalizadas, investigar o motivo.
    Resposta : A dimensão do fluxo depende do autor dos dados, logo autores diferentes podem já normalizar os dados.

Problema : A normalização não está funcionando, acusa que é um tipo de escalar não iterável.
    Resposta : Era um problema que foi corrigida na versão mais recente do Astropy (6.0.1). Resolvendo isso,
    foi possível normalizar a curva tanto pelo .stich() como pelo .normalize().
    
Problema : A variação no fluxo de luz está invertido (variação positiva ao invés de negativa), isso é um problema?
    Resposta : Sim, era um problema, alguma curvas estavam vindo normalizadas da importação direta, como posteriormente
    seria feita uma nova normalização, essas curva já normalizada interferiam no processo de normalização secundário. 
    Por esse motivo importa-se somente curvas não normalizadas.

Problema : Como eu farei o seccionamento das curvas de luz para sobrepor elas? Pois a estrutura dos dados não
carrega consigo somente o fluxo. Mesmo que eu cortasse somente os fluxo em períodos bem definidos iria plotar em
função do que?
    Resposta : Quando usa-se dataframa, é possível dividir fazer tudo isso sem a necessidade de desmebrar o objeto, basta 
    dividir a coluna do tempo pelo resto da divisão do período.

Problema : A função to_pandas() está tornando a culuna do tempo como um índice (SUPOSTAMENTE), isso é problemático pois não
consigo trabalhar com os valores de tempo.

"""

#%%
"""
Bibliotecas a serem importadas:
    * lightkurve : https://docs.lightkurve.org/whats-new-v2.html
    * numpy : https://numpy.org/doc/
    * matplotlib.pyplot :  https://matplotlib.org/stable/index.html

"""
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
"""
Variáveis de controle.

"""
SHOW_LC_PLOT = False
SHOW_LC_INFORMATION = False

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
                                      author = 'SPOC',  # Cada autor usa uma grandeza de fluxo  
                                      sector = (list(range(1, 3)))  # quarter|sector|campaign
                                      )

if SHOW_LC_INFORMATION:
    print(search_result) # Mostra a tabela com as informações da curva de luz 

#%%
"""
Dados do TIC 38846515:
    * Planet Transit Duration [hours] : 3.567 ± 0.053
    * Planet Orbital Period [days] : 2.8493825 ± 0.0000003
    * Planet Radius [R_Earth] : 16.3869 ± 0.734704
"""


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
#lc = lc_collection.stitch(corrector_func=None)
lc_normal = lc_collection.stitch() 

#%%
if SHOW_LC_PLOT:
    lc_normal.plot() # Plot com curvas de luz normalizadas 

#%%
"""
Outra maneira de normalizar a curva pela função 
    .normalize()
porém o stich parece melhor.
"""
#lc_normalize = lc_collection[0].normalize() 

#%%
"""
Uma maneira de realizar a dobra com superposição da curva de luz é utilizando a função 
    .fold(perido,tempo inicial)

"""
period = 2.8493825 # perído orbital do exoplaneta em dias 
time_inicial = lc_normal.time.value[0] # tempo incial da curva de luz
lc_superpostion = lc_normal.fold(period, time_inicial - 1.4) # -1.4 para centralizar
lc_superpostion.scatter()
plt.xlim(-0.4, 0.4)
plt.show()


#%%
"""
Método para trabalho com as colunas do objeto 'lightkurve.lightcurve.TessLightCurve' em forma de array.

"""
df_aux = lc_normal.to_pandas()
time = (df_aux.index % period)
flux = df_aux.flux.to_numpy()
df_LC = pd.DataFrame({'time': time , 'flux': flux})
df_LC = df_LC.sort_values(by='time')
#print(df_LC)

#%%
plt.scatter(df_LC['time'], df_LC['flux'], s=2)
plt.xlim(1.6, 1.95)
plt.show()














