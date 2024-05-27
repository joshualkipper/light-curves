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
Exemplos que dão certo.

    Dados do TIC 38846515:
        * Planet Transit Duration [hours] : 3.567 ± 0.053
        * Planet Orbital Period [days] : 2.8493825 ± 0.0000003
        * Planet Radius [R_Earth] : 16.3869 ± 0.734704
        * Star Radius [R_Sun] : 1.77 ± 0.08
        * Sectors : 1,2,3,4,5,6,7,8,9,10,11,12,13,27,28,29,30,31,32,33,34,35,36,37,38,39,61,62,63,64,65,67,68,69 
        * Author : SPOC
        
    Dados do TIC 231663901:
        * Planet Transit Duration [hours] : 1.6438728317153 ± 0.018756187
        * Planet Orbital Period [days] : 1.4303691403398 ± 9.228199e-7	
        * Planet Radius [R_Earth] : 13.250493344819 ± 0.66068995
        * Star Radius [R_Sun] : 0.89077401161194 ± 0.0438467
        * Sectors : 1,27
        * Author : SPOC
        
    Dados do TIC 48451130:
        * Planet Transit Duration [hours] : 6.0823745326155 ± 0.25691837
        * Planet Orbital Period [days] : 30.3599811786 ± 0.00031322188
        * Planet Radius [R_Earth] : 11.214121596787 ± 0.4613171
        * Star Radius [R_Sun] : 0.79511797428131
        * Sectors : 40,41,54,55
        * Author : SPOC

"""
"""
Exemplos que dão errado.
    
    Dados do TIC 27769688:
        * Planet Transit Duration [hours] : 6.95108488405611 ± 0.39727047
        * Planet Orbital Period [days] : 88.4956829658291 ± 0.010978528
        * Planet Radius [R_Earth] : 5.15743025745616 ± 0.5870564
        * Star Radius [R_Sun] : 0.939951002597809 ± 0.049598
        * Sectors : 14,15,40,41,54,55
        * Author : SPOC
    Problema : não se observa nenhum trânsito.

    Dados do TIC 158657354:
        * Planet Transit Duration [hours] : 11.0403764815672 ± 0.71893275
        * Planet Orbital Period [days] : 38.5835888603263 ± 0.0011334544	
        * Planet Radius [R_Earth] : 6.88722761627487 ± 1.0315099
        * Star Radius [R_Sun] : 1.74231004714966 ± 0.074485
        * Sectors : 40,41,53,54,55
        * Author : SPOC
    Problema : não se observa nenhum trânsito.


    Dados do TIC 299087490:
        * Planet Transit Duration [hours] : 5.1869736832286 ± 0.23243478
        * Planet Orbital Period [days] : 38.478761309913 ± 0.003233638
        * Planet Radius [R_Earth] : 12.560937578888 ± 0.5219885
        * Star Radius [R_Sun] : 0.81004297733307
        * Sectors : 40,41,53,54,55
        * Author : SPOC
    Problema : Mesmo após a superposição das curvas observase mais de um trânsito
"""
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
"""
Variáveis de controle.

"""
SHOW_LC_PLOT = True
SHOW_LC_INFORMATION = True
ALTERNATIVE_METHOD = True

#%%

"""
Função utilizada para pesquisar as curvas de luz disponíveis 
    * lk.search_lightcurve(target, radius=None, exptime=None, 
                            cadence=None, mission=('Kepler', 'K2', 'TESS'), 
                            author=None, quarter=None, month=None, campaign=None, 
                            sector=None, limit=None)[source]

"""
search_result = lk.search_lightcurve('TIC 158657354',    # ID do alvo
                                      cadence ='short',     # ‘long’|‘short’|‘fast'|float
                                      mission = 'TESS',     # Missão autora dos dados
                                      author = 'SPOC',      # Cada autor usa uma grandeza de fluxo  
                                      sector = (40,41,53,54,55)    # quarter|sector|campaign
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
period = 38.5835888603263 # perído orbital do exoplaneta em dias
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

time_transit = 11.0403764815672 # em horas
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