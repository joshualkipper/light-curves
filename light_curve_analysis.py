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
import os

#%%
"""
Informações do exoplaneta que será analisado.
"""
exoplanet = {
    'TIC_ID' : 190998418,
    'period' : 2.87929997, # Dias
    'time_transit' : 2.7497683357845, # Horas
    'sectors': (17,57)  
    }
#%%
"""
Variáveis de controle.
"""
SHOW_PLOT = True
SHOW_INFORMATION = False
SHOW_ALTERNATIVE_METHOD = False
DOWNLOAD_PLOT = False

#%%

"""
Função utilizada para pesquisar as curvas de luz disponíveis.
    * lk.search_lightcurve()
"""
search_result = lk.search_lightcurve(f'TIC {exoplanet["TIC_ID"]}',    # ID do alvo
                                      cadence ='short',     # ‘long’|‘short’|‘fast'|float
                                      mission = 'TESS',     # Missão autora dos dados
                                      author = 'SPOC',      # Cada autor usa uma grandeza de fluxo  
                                      sector = exoplanet['sectors']    # quarter|sector|campaign
                                      )

if SHOW_INFORMATION:
    print(search_result) # Mostra a tabela com as informações da pesquisa

#%%
"""
Função para fazer download de todas as curvas de luz encontradas. 
    *.download_all() 
"""
lc_collection = search_result.download_all()
if SHOW_INFORMATION:
   print(lc_collection)
   print(f"Número de curvas de luz: {len(lc_collection)}")

#%%
"""
Cada Curva de Luz tem a seguinte estrutura do tipo "dataframe".
"""
if SHOW_INFORMATION:
    # O algarismo vai de 0 até N-1 onde N é o número de curvas de luz
    print(lc_collection[0]) 
#%%
if SHOW_INFORMATION:
    # Exibe as colunas do objeto lightkurve
    print(lc_collection[0].columns) 

#%%
"""
Função para visualizar o gráfico com as curvas de luz.
    .plot()     
"""
if SHOW_PLOT:
    plt.figure(figsize = (10,5))
    lc_collection.plot() 
    plt.legend(fontsize = 0)
    plt.title('Light Curve Collection', fontsize = 16)
    plt.show()

#%%
"""
Função para combinar todas as curvas de luz em uma única curva de luz.
    .stitch() 
o qual retorna um objeto da classe lightkurve.lightcurve.TessLightCurve. O comando sem 
argumentos irá retornar uma curva normalizada de todas da união de todas as curvas.
"""
if SHOW_ALTERNATIVE_METHOD:
    lc = lc_collection.stitch(corrector_func=None)

lc_aux = lc_collection.stitch() 
lc_normal = lc_aux.remove_nans() # Removendo 'NANS' para não atrapalhar nas análises

#%%
if SHOW_PLOT:
    f = lc_normal.flux
    t = lc_normal.time.value
    fig, axs = plt.subplots(figsize = (10,5), dpi = 200)
    text_legend = (
        f'TIC ID : {exoplanet["TIC_ID"]}\n'
        f'Period : {exoplanet["period"]:.2f} Days\n'
        f'Transit : {exoplanet["time_transit"]:.2f} Hours'
    )
    axs.scatter(t, f, s = 1, label = text_legend, color = 'indigo')
    axs.set_title('Light Curve Collection Normalized', fontsize = 16)
    axs.legend(fontsize = 11, edgecolor = 'black')
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
    plt.show()

#%%
"""
Outra maneira de normalizar a curva pela função
    .normalize()
porém o stich parece melhor.
"""
if SHOW_ALTERNATIVE_METHOD:
    lc_normalize = lc_collection[0].normalize() # Normaliza somente uma curva
    if SHOW_PLOT:
        t = lc_normalize.time.value
        f = lc_normalize.flux
        fig, axs = plt.subplots(figsize = (10,5), dpi = 200)
        text_legend = (
            f'TIC ID : {exoplanet["TIC_ID"]}\n'
            f'Period : {exoplanet["period"]:.2f} Days\n'
            f'Transit : {exoplanet["time_transit"]:.2f} Hours\n'
            f'Sector : {lc_collection.sector[0]}'
        )
        axs.scatter(t, f, s = 1, label = text_legend, color = 'indigo')
        axs.set_title('Light Curve Normalized', fontsize = 16)
        axs.legend(fontsize = 11, edgecolor = 'black')
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
        plt.show()

#%%
"""
Aqui são plotados todas as curvas do 'lc_collection' normalizados um de cada vez.
"""
# Exibir todos os gráficos de todas as curvas de luz
if SHOW_ALTERNATIVE_METHOD:
    # Obter a lista de cores padrão do matplotlib
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (lc, color) in enumerate(zip(lc_collection, default_colors)):
        lc_normalize = lc.normalize()  # Normaliza a curva atual
        if SHOW_PLOT:
            t = lc_normalize.time.value
            f = lc_normalize.flux
            fig, axs = plt.subplots(figsize=(10, 5), dpi=200)
            text_legend = (
                f'TIC ID : {exoplanet["TIC_ID"]}\n'
                f'Period : {exoplanet["period"]:.2f} Days\n'
                f'Transit : {exoplanet["time_transit"]:.2f} Hours\n'
                f'Sector : {lc_collection.sector[i]}'
            )
            axs.scatter(t, f, s=1, label=text_legend, color=color)
            axs.set_title('Light Curve Normalized', fontsize=16)
            axs.legend(fontsize=11, edgecolor='black')
            # Configurações dos eixos e da borda
            for spine in axs.spines.values():
                spine.set_color('black')  # Cor da borda
                spine.set_linewidth(1)  # Largura da borda
            axs.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))  # N° de risquinhos em x
            axs.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))  # || || || em y
            # Ajustando o tamanho, cor e direção dos risquinhos
            axs.tick_params(which='minor', length=5, color='black', direction='in')
            axs.tick_params(which='major', length=8, color='black', direction='in')
            axs.tick_params(axis='both', labelsize=12)
            axs.set_xlabel("Phase [Days]", fontsize=12)
            axs.set_ylabel("Normalized Flux", fontsize=12)
            plt.show()

#%%
"""
Método para trabalho com as colunas do objeto 'lightkurve.lightcurve.TessLightCurve' em 
forma de array. Esse método é extremamente manual, por algumas particularidade do lightkurve
é necessário converter o objeto para um dataframe do pandas e ai trabalhar com ele.
"""
if SHOW_ALTERNATIVE_METHOD:
    # Temos que converter o objeto lightkurve em um dataframe do pandas pois não 
    # conseguimos trabalhar com ele
    df_aux = lc_normal.to_pandas()
    time = (df_aux.index % exoplanet['period']) # Maneira de dobrar as curva de luz
    flux = df_aux.flux.to_numpy() # Remove o tempo como índice 
    df_LC = pd.DataFrame({'time': time , 'flux': flux}) # Criando um novo dataframe
    df_LC = df_LC.sort_values(by='time') # Reordenando de maneira crescente no tempo
    if SHOW_INFORMATION:
        print(df_LC)
    if SHOW_PLOT:
        t = df_LC['time']
        f = df_LC['flux']
        fig, axs = plt.subplots(figsize = (10,5), dpi = 200)
        text_legend = (
            f'TIC ID : {exoplanet["TIC_ID"]}\n'
            f'Period : {exoplanet["period"]:.2f} Days\n'
            f'Transit : {exoplanet["time_transit"]:.2f} Hours'
        )        
        axs.scatter(t, f, s = 1, label = text_legend, color = 'indigo')
        axs.set_title('Light Curve Superposition', fontsize = 16)
        axs.legend(fontsize = 11, edgecolor = 'black')
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
        plt.show()

#%%
"""
Função para realizar a sobreposição das curvas de luz em um único trânsito.
    *.fold()
"""
time_inicial = lc_normal.time.value[0] # Tempo inicial da curva de luz
# Em geral esses dados estão descentralizados
lc_fold = lc_normal.fold(exoplanet['period'], time_inicial) 
# Alteracoes para conseguir usar o plt.plot()
if SHOW_PLOT:
    t = lc_fold.time.value 
    f = lc_fold.flux
    # Constante para seccionar o transito no tempo adequeado 
    fig, axs = plt.subplots(figsize = (10,5), dpi = 200)
    text_legend = (
        f'TIC ID : {exoplanet["TIC_ID"]}\n'
        f'Period : {exoplanet["period"]:.2f} Days\n'
        f'Transit : {exoplanet["time_transit"]:.2f} Hours'
    )        
    axs.scatter(t, f, s = 1, label = text_legend, color = 'indigo')
    axs.set_title("Off-Center Light Curve Superposition", fontsize = 16)
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
if SHOW_ALTERNATIVE_METHOD:
    flux_aux = lc_fold.flux
    time_aux = lc_fold.time
    time_min = time_aux.value[flux_aux.argmin()]
    lc_superposition = lc_normal.fold(exoplanet['period'], time_inicial + time_min)
    if SHOW_PLOT:
        # Alteracoes para conseguir usar o plt.plot()
        t = lc_superposition.time.value 
        f = lc_superposition.flux
        # Constante para seccionar o transito no tempo adequeado 
        section = (exoplanet['time_transit'] / 24) * 2 
        fig, axs = plt.subplots(figsize = (10,5), dpi = 200)
        text_legend = (
            f'TIC ID : {exoplanet["TIC_ID"]}\n'
            f'Period : {exoplanet["period"]:.2f} Days\n'
            f'Transit : {exoplanet["time_transit"]:.2f} Hours'
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
        axs.set_xlim(-section, section) # Corte no eixo temporal
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
"""
Formatação dos dados para que seja possível iterá-los. Além disso escrevemos a matriz
em ordem crescente, em relação ao fluxo, isso nos permite fugir da faixa centrar onde 
o fluxo é constante, e portanto seriam satisfeitas as condições de vizinhos.
"""
time_column = lc_fold.time.value.tolist()
flux_column = lc_fold.flux.value.tolist()
matrix = np.column_stack((time_column, flux_column))
# Escrevendo od indíces em ordem crescente em relação ao fluxo
new_index = np.argsort(matrix[:, 1])  
points = matrix[new_index] # Uma matriz Nx2, com os pontos de interesse

#%%
"""
Função que tem como objetivo determinar o valor de ε utilizando os dados de fluxo. 
Utilizamos a diferenças entre o fluxo médio e o fluxo minimo para estimar a ordem do
raio da vizinhança.
"""
def epslon(flux_medium, flux_minimun):
    a = flux_medium - flux_minimun
    if a == 0:
        return 0
    # Função que tira o expoente de 'a' e pega o valor inteiro do produto com o log
    exponent = mt.floor(mt.log10(abs(a)))
    epslon = 1 * (10**exponent)
    return epslon

#%%
flux_medium = lc_fold.flux.mean()
flux_minimun = lc_fold.flux.min()
# Estimativa do raio da vizinhança do ponto mínimo
radius = epslon(flux_medium, flux_minimun) 
# Novo ajuste para o tempo tal que o fluxo seja minímo
new_time_min = neighborhood(points, radius)

#%%
"""
Segunda tentativa de centralizar os dados usando a vizinhança do ponto minímo.
"""
lc_superposition = lc_normal.fold(exoplanet['period'], time_inicial + new_time_min[0])
# Gráfico com foco no transito superposto
for i in range(3):
    constant_time = [1, 1, 24] # Lista para mudança da fase
    # Alteracoes para conseguir usar o plt.plot()
    t = lc_superposition.time.value * constant_time[i]
    f = lc_superposition.flux
    # Constante para seccionar o transito no tempo adequeado 
    section = (exoplanet['time_transit'] / 24) * 2 * (constant_time[i])
    fig, axs = plt.subplots(figsize = (10,5), dpi = 200)
    text_legend = (
        f'TIC ID : {exoplanet["TIC_ID"]}\n'
        f'Period : {exoplanet["period"]:.2f} Days\n'
        f'Transit : {exoplanet["time_transit"]:.2f} Hours'
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
    text_time = ['Days', 'Days', 'Hours']
    axs.set_xlabel(f"Phase[{text_time[i]}]", fontsize = 12)
    axs.set_ylabel("Normalized Flux", fontsize = 12)
    axs.set_ylim(flux_minimun - radius/10, abs(flux_medium-flux_minimun) + flux_medium + radius/10)
    if i != 0:
        axs.set_xlim(-section, section) # Corte no eixo temporal
    if DOWNLOAD_PLOT:
        name = [f'lc(TIC_ID:{exoplanet["TIC_ID"]}).png',
                f'lc_section_days(TIC_ID:{exoplanet["TIC_ID"]}).png',
                f'lc_section_hours(TIC_ID:{exoplanet["TIC_ID"]}).png']
        path_base = '/home/joshua/Documentos/iniciacao_cientifica/light-curves/examples'
        new_directory = os.path.join(path_base, f'TIC_ID:{exoplanet["TIC_ID"]}')
        os.makedirs(caminho_pasta, exist_ok=True)
        path_plots = os.path.join(new_directory, name[i])
        plt.savefig(path_plots)
    plt.show()

#%%
