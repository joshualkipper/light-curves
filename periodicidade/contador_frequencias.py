# %%
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.signal import find_peaks

# %% [Configurações iniciais do exoplaneta]
exoplanet = {
    'TIC_ID': 25155310,
    'period': 3.28878982693448,         # dias
    'time_transit': 3.43677173022605,   # horas 
    'sector': 1
}

# %% [Download e pré-processamento da curva de luz]
search = lk.search_lightcurve(f'TIC {exoplanet["TIC_ID"]}',
                              cadence='short',
                              mission='TESS',
                              author='SPOC',
                              sector=exoplanet['sector'])

lc = search.download_all().stitch().remove_nans().normalize(unit='ppm') - 1

# %% [Análise de Fourier]
# Define o intervalo de busca para o período
max_freq = 20  # 1/dia
min_period = round(1 / max_freq, 2)
max_period = 20

# Geração do periodograma de potência
pg = lc.to_periodogram(minimum_period=min_period, maximum_period=max_period)
periodo_dominante = pg.period_at_max_power.value
periodo_real = exoplanet["period"]

# %% [Plot: curva de luz normalizada]
lc.plot(title='Curva de Luz Normalizada')
plt.show()

# %% [Plot: periodograma em função do período]
ax = pg.plot(title='Periodograma em função do Período')
ax.axvline(periodo_real, color='r', linestyle='--', label=f'Período real: {periodo_real:.5f} d')
ax.axvline(periodo_dominante, color='g', linestyle='-.', label=f'Período Fourier: {periodo_dominante:.5f} d')
ax.legend(loc='best')
ax.set_xlim(-0.1, 5)
plt.show()

# %% [Conversão para domínio da frequência]
freqs = 1 / pg.period.value
powers = pg.power.value

# %% [Parâmetros da varredura por picos]
delta_f = 1 / exoplanet["period"]   # largura da janela
N_sigma = 2.5                       # fator de significância
max_skip = 2                        # janelas consecutivas sem pico

# Inicialização da varredura
f_atual, f_max = freqs.min(), freqs.max()
picos_freq, picos_power = [], []
skip_count = 0

# %% [Identificação de picos significativos]
while f_atual < f_max:
    # Seleção de dados na janela atual
    janela_mask = (freqs >= f_atual) & (freqs < f_atual + delta_f)
    f_janela = freqs[janela_mask]
    p_janela = powers[janela_mask]

    # Continua somente se houver dados suficientes
    if len(f_janela) > 5:
        media, desvio = np.mean(p_janela), np.std(p_janela)
        limiar = media + N_sigma * desvio

        # Identificação dos picos locais
        indices_pico, _ = find_peaks(p_janela)
        if indices_pico.size:
            idx_max = indices_pico[np.argmax(p_janela[indices_pico])]
            pico_power = p_janela[idx_max]

            # Armazena se ultrapassar o limiar
            if pico_power > limiar:
                picos_freq.append(f_janela[idx_max])
                picos_power.append(pico_power)
                skip_count = 0
            else:
                skip_count += 1
        else:
            skip_count += 1
    else:
        skip_count += 1

    if skip_count >= max_skip:
        print(f"Análise encerrada: {max_skip} janelas consecutivas sem pico significativo.")
        break

    f_atual += delta_f

# %% [Plot: periodograma em função da frequência]
plt.figure(figsize=(10, 5), dpi=200)
plt.plot(freqs, powers, label=f'TIC {exoplanet["TIC_ID"]}')
plt.axvline(1 / periodo_real, color='r', linestyle='--', label=f'Freq. real: {1 / periodo_real:.5f} 1/d')
plt.axvline(1 / periodo_dominante, color='g', linestyle='-.', label=f'Freq. Fourier: {1 / periodo_dominante:.5f} 1/d')
plt.scatter(picos_freq, picos_power, color='orange', marker='x', zorder=5, label='Harmônicos')
plt.xlabel("Frequência (1/dia)")
plt.ylabel("Potência")
plt.title("Periodograma em função da Frequência")
plt.grid(True)
plt.legend()
plt.show()

# %% [Resumo dos resultados]
print(f"Período dominante (Fourier): {periodo_dominante:.5f} dias")
print(f"Período real do exoplaneta: {periodo_real:.5f} dias")
print(f"Número de picos detectados: {len(picos_freq)}")

