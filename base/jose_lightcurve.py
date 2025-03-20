#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PURPOSE:
    
    SÉRIES TEMPORAIS são sequências de medidas de alguma grandeza Y realizadas 
    ao longo do tempo t, ordenadas em ordem cronológica: {t_i, y_i}, i=1,n,
    t_i < t_(i+1). As medidas podem estar igualmente espaçadas no tempo ou
    não. 
    
    Por exemplo, medidas do número máximo de manchas solares observadas em 
    cada ano, ao longo de vários anos; medidas das variações do brilho de 
    uma estrela ao longo de uma noite; medidas do nível de precipitação de
    águas das chuvas ao longo dos meses; etc.
    
    A ANÁLISE DE SÉRIES TEMPORAIS permite extrair estatísticas úteis sobre
    uma série temporal que nos permitem saber SE a grandeza medida realmente
    varia no tempo e COMO estas variações ocorrem.
    
    Um exemplo particular de séries temporais são as CURVAS DE LUZ de objetos
    astronômicos. Uma curva de luz é uma sequência de medidas fotométricas 
    do brilho de um objeto, geralmente, em uma banda espectral específica,
    ao longo de um certo intervalo de tempo.
    
    Nesta aula, usaremos a biblioteca "lightkurve" do Python para analisar a
    curva de luz de uma estrelas variável observada pelo Telescópio Espacial
    KEPLER.
    
    Talvez você tenha que instalar o módulo "lightkurve" em seu computador:
        
        * como root:
            % conda install lightkurve    ou
            % pip install lightkurve
            
        * como usuário comum - neste caso, convém criar um ambiente usando
          o Conda. 
            % conda create --name env1 lightkurve spyder
            
            % conda activate env1      # para ativar o ambiente
            OU                         #
            % conda source env1        # .. OU ...
            ...
            % conda deactivate         # para desativar o ambiente
        
    Fonte / documentação sobre o "lightkurve":
        * https://docs.lightkurve.org/whats-new-v2.html
    
"""

# Importando os módulos necessários:
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt

#%% ###########################################################################
# 1. Obtendo os dados diretamente do banco de dados do MAST data archive.
#
#    Neste exemplo, vamos analizar a curva de luz da estrela "KIC 10963065",
#    a qual é uma estrela variável de tipo solar. 
#
#    O MAST (Mikulski Archive for Space Telescope) é um centro projetado para
#    hospedar dados de várias missões espaciais como Hubble, Kepler, TESS, 
#    James Web, etc, inclusive da futura missão Roman, que realizam observações
#    na região óptica do espectro (visível, UV e IR). O MAST facilita 
#    enormemente o acesso aos dados dessas missões.
#
# #############################################################################


# 1.1 - Pesquisando as curvas de luz disponíveis:
#
#    A função "lk.search_lightcurve(target, ...)" permite procurar por curvas
#    de luz do objeto-alvo (target) ou de qualquer objeto ao redor dele
#    dentro de um certo raio, se o argumento do parâmetro "radius" for
#    especificado.
#
#    lk.search_lightcurve(target, radius=None, exptime=None, 
#                         cadence=None, mission=('Kepler', 'K2', 'TESS'), 
#                         author=None, quarter=None, month=None, campaign=None, 
#                         sector=None, limit=None)[source]
#    
#    Esta função retorna um objeto (que não é um pd.DataFrame), mas tem uma
#    estrutura de tabela e é legível, contendo uma listagem de todas as curvas
#    de luz disponíveis e método 
#
#    Neste exemplo, vamos pesquisar por todas as curvas de luz da estrela
#    KIC 10963065, obtidas pela missão Kepler, em cadência rápida ('short')
#    em 4 'quarters':

search_result = lk.search_lightcurve('KIC 10963065',    # ID do alvo
                                     cadence='short',   # ‘long’|‘short’|‘fast'|float
                                     author='Kepler',   # missão autora dos dados
                                     quarter=(2,5,6,7)  # quarter|sector|campaign
                                     )

print(search_result)

#%% 1.2 - Fazendo download das curvas de luz:
#
#    Para fazer download de todas as curvas de luz, use o método .download_all()
#    o qual retorna um objeto da classe lk.collections.LightCurveCollection
#    contendo uma coleção com a curva de luz para cada quarter.

lc_collection = search_result.download_all() #.stitch()

print(lc_collection)
print(f"Número de curvas de luz: {len(lc_collection)}")



#%% Cada curva de luz, dentro desta coleção, é um objeto onde os dados têm
#   uma estrutura de tabela com as seguintes colunas (no caso do Kepler)
#
#   'time','flux','flux_err','quality','timecorr','centroid_col','centroid_row',
#   'cadenceno','sap_flux','sap_flux_err','sap_bkg','sap_bkg_err','pdcsap_flux',
#   'pdcsap_flux_err','sap_quality','psf_centr1','psf_centr1_err','psf_centr2',
#   'psf_centr2_err','mom_centr1','mom_centr1_err','mom_centr2','mom_centr2_err',
#   'pos_corr1','pos_corr2')

print(lc_collection[2])



#%% Para checar os nomes das colunas:
print(lc_collection[2].columns)


#%% Para VISUALIZAR a coleção de curvas de luz,
#   utilize o método .plot() 
#
#   O gráfico mostra as medidas do fluxo fotométrico (em unidades de e-/s) em
#   função do tempo, expresso em data juliana (JD) para um referencial
#   no baricentro (centro de massa) do Sistema Solar (BJD). 
#
#   A data juliana (JD) de um evento é o número de dias solares (com duração de 24h), 
#   transcorridos desde às 12h00 de 1/Janeiro/4713 aC até o instante de
#   ocorrência do evento. Horas, minutos e segundos são expressos como frações
#   de um dia juliano.
#
#   Datas julianas são números grandes, com muitos dígitos, por exemplo,
#   para 14h00 (UT1) de 07/02/2024, JD = 2460342.590856, o que torna imprático
#   o seu uso. Por isto, costuma-se utilizar uma forma "reduzida" da data 
#   juliana, pela subtração de um número inteiro, arbitrário, de dias. 
#   No caso do Kepler, a data resultante é referida como "Kepler Julian Date" (KJD). 
#
#   O instante de tempo t em que a luz de um alvo chega no telescópio espacial 
#   Kepler, depende da posição orbital do satélite, o que é altamente 
#   inconveniente. Para tornar as medidas de tempo independentes da posição
#   orbital do satélite, calcula-se o instante de tempo t' em que o mesmo
#   sinal seria observado no baricentro (centro de massa) do Sistema Solar,
#   resultando no "Barycentric Kepler Julian Date" (BKJD) usado nas curvas
#   de luz.
#
lc_collection.plot()

#%% Para COMBINAR todas as curvas de luz em uma única curva de luz,
#   utilize o método .stitch(), o qual retorna um objeto da classe
#   lk.lightcurve.KeplerLightCurve  (no caso do Kepler). 
#
#   Este método possui o parâmetro "corrector_func" para o qual se pode
#   passar como argumento uma função Python para fazer as "correções"
#   ou tratamento desejado nas curvas de luz antes de concatenálias.
#
#   Se nenhuma função for passada como argumento, por default, para cada 
#   curva de luz individual, o método .stitch() fará (1) sua normalização,
#   isto é, dividirá as medidas de fluxo pelo fluxo médio e, em seguida,
#   o "achatamento" ("flattening"), ajustando um polinômio de baixa ordem
#   e dividindo cada medida do fluxo pelo valor do polinômio. Desta forma,
#   os fluxos medidos ficarão dispersos ao redor da unidade. 
#
#   As curvas de luz tratadas são então concatenadas em uma série temporal.
#
#   Além dos dados da curva de luz, este objeto possui vários métodos para 
#   a realização de operações com a curva de luz e para a análise
#   da mesma.

lc = lc_collection.stitch(corrector_func=None)
#lc = lc_collection.stitch()

#%% Para visualizar a curva de luz total:
#   
lc.plot()




#%% ###########################################################################
# 2. Procurando por periodicidades na curva de luz.
#
#    O método .to_periodogram() retorna um "PERIODOGRAMA" para a curva de luz,
#    calculado conforme um dados método. Há diferentes métodos numéricos que 
#    podem ser usados no cálculo de periodgramas, mas o método .to_periodogram()
#    oferece duas opções para o parâmetro "method":
#
#       * Lomb-Scargle         (method = "lombscargle" ou "ls")
#       * Box Least-Squares    (method = "boxleastsquares" ou "bls")
# #############################################################################

#%% periodogramas de Lomb-Scargle: freq x Power
pg = lc.to_periodogram(method='lombscargle',
                       normalization='amplitude',  #default
                       minimum_frequency=1500,
                       maximum_frequency=2700)

pg.plot();

#%% extraindo as frequências e potência 
freq = pg.frequency
power = pg.power

plt.plot(freq, power, '-k', lw='0.2')
plt.xlabel(f"Frequency ($\mu$Hz)")
plt.ylabel("Power")



#%% frequencia x amplitude
amplitude = np.sqrt(power) 

plt.plot(freq, amplitude, '-k', lw='0.2')
plt.xlabel(f"Frequency ($\mu$Hz)")
plt.ylabel("Amplitude (ma)")


#%% ===========================================================================
#   periodogramas de Lomb-Scargle: freq x Power : 
pg = lc.to_periodogram(method='lombscargle',
                       normalization='psd',
                       minimum_frequency=1500,
                       maximum_frequency=2700)

pg.plot()
plt.show()


#%% ===========================================================================
# Gerando um periodograma suavizado:
ax = pg.plot(label='Original spectrum')

pg.smooth(filter_width=1).plot(ax=ax,
                               color='green',
                               label='Smoothed spectrum')



# Plotando um "envelope" para os modos de pulsação, usando uma curva Gaussiana
f = pg.frequency.value

ax.plot(f, 5e-11*np.exp(-(f-2100)**2/(2*230**2)),
        lw=2, ls='--', zorder=0,
        color='blue', label='Mode envelope')


# Indicando a posição dos modos de pulsação com flexas vermelhas
for i in range(6):
  ax.annotate('',
              xy=(1831.66+i*103.8, 5.2e-11),
              xytext=(1831.66+i*103.8, 7e-11),
              arrowprops=dict(arrowstyle='->',
              color='red',
              linewidth=1.5))
  
ax.legend();



