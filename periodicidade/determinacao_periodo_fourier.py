#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste para a determinação do período de uma curva de luz via transformada de
fourier.
"""
#%%
import lightkurve as lk # versão 2.4.2
import matplotlib.pyplot as plt # versão 3.5.1
#%%
exoplanet_1 = {
    'TIC_ID' : 25155310,
    'period' : 3.28878982693448, # Dias
    'time_transit' : 3.43677173022605, # Horas
    'sectors': (1,2)  
    }

exoplanet_2 = {
    'TIC_ID' : 238176110,
    'period' : 2.79858018416552, # Dias
    'time_transit' : 2.38010434405069, # Horas
    'sectors': (1,27)  
    }


#%%
search_result = lk.search_lightcurve(f'TIC {exoplanet_2["TIC_ID"]}',    # ID do alvo
                                      cadence ='short',     # ‘long’|‘short’|‘fast'|float
                                      mission = 'TESS',     # Missão autora dos dados
                                      author = 'SPOC',      # Cada autor usa uma grandeza de fluxo  
                                      sector = exoplanet_1['sectors']    # quarter|sector|campaign
                                      )

lc_collection = search_result.download_all()
lc_aux = lc_collection.stitch() 
lc_normal = lc_aux.remove_nans() # Removendo 'NANS' para não atrapalhar nas análises

#%%
lc_fourier = lc_normal.to_periodogram(
                       minimum_period=1.0,   # mínimo de 1 dia
                       maximum_period=10.0)  # máximo de 10 dias
periodo_dominante = lc_fourier.period_at_max_power.value
periodo_real = list({exoplanet_2["period"]})[0]
#%%
lc_normal.plot()
ax = lc_fourier.plot()  # sem o label aqui

ax.axvline(periodo_real, color='r', linestyle='--',
          label=f'Período real: {periodo_real:.5f} d')

ax.axvline(periodo_dominante, color='g', linestyle='-.',
          label=f'Período Fourier: {periodo_dominante:.5f} d')

ax.legend(loc='best')
plt.show()
#%%
print(f"Período dominante: {lc_fourier.period_at_max_power}\n"
      f"Período real: {exoplanet_1['period']} d")