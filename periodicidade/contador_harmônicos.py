# %%
"""
Script para análise automática de curvas de luz de exoplanetas usando dados do TESS.
Este código:
- Faz download da curva de luz de um alvo TESS a partir de um CSV de alvos.
- Constrói o periodograma para identificar frequências e harmônicos associados ao período orbital.
- Detecta automaticamente quantos harmônicos estão presentes, adaptando dinamicamente a faixa de frequências analisada.
- Gera gráficos da curva de luz e do periodograma, com os harmônicos destacados.
- Pode salvar os gráficos em disco.
"""
# %% Importações
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.signal import find_peaks

# %% Funções auxiliares

def baixar_curva_luz(tic_id, setores):
    """
    Baixa e prepara a curva de luz para um TIC específico e uma lista de setores.
    Remove NaNs, normaliza e ajusta a unidade para ppm.
    """
    if isinstance(setores, int):
        setores = [setores]

    curvas = []
    for setor in setores:
        resultado = lk.search_lightcurve(f'TIC {tic_id}', cadence='short',
                                         mission='TESS', author='SPOC', sector=setor)
        if resultado:
            try:
                curva = (resultado.download_all()
                         .stitch()
                         .remove_nans()
                         .normalize(unit='ppm') - 1)
                curvas.append(curva)
            except Exception as e:
                print(f"⚠️ Falha no setor {setor} para TIC {tic_id}: {e}")
    
    if curvas:
        return curvas[0].append(curvas[1:]) if len(curvas) > 1 else curvas[0]
    return None


def identificar_harmonicos(freqs, powers, freq_base, limiar_percentil, tolerancia, max_harm=50):
    """
    Identifica harmônicos com base na frequência base e um limiar de potência (percentil).
    Considera uma tolerância para reconhecer harmônicos fora do valor exato esperado.
    """
    limiar = np.percentile(powers, limiar_percentil)
    indices, _ = find_peaks(powers, height=limiar)

    picos_freq = freqs[indices]
    picos_pot = powers[indices]
    harmonicos = []

    for n in range(1, max_harm + 1):
        alvo = n * freq_base
        dentro_faixa = ((picos_freq > alvo * (1 - tolerancia)) & 
                        (picos_freq < alvo * (1 + tolerancia)))
        candidatos = picos_freq[dentro_faixa]

        if candidatos.size > 0:
            melhor_idx = np.argmax(picos_pot[dentro_faixa])
            harmonicos.append((candidatos[melhor_idx], picos_pot[dentro_faixa][melhor_idx]))

    return harmonicos


def detectar_harmonicos_com_expansao(
    curva,
    freq_base,
    max_harm_inicial=5,
    harm_step=15,
    harm_limite=100,
    limiar_percentil=90.0,
    tolerancia=0.03,
    oversample=15
):
    """
    Detecta harmônicos dinamicamente, expandindo a faixa de frequências conforme novos picos são encontrados.
    A expansão para quando não há mais novos harmônicos detectados.
    """
    max_harm = max_harm_inicial
    harmonicos_previos = []

    while max_harm <= harm_limite:
        max_freq = max_harm * freq_base
        pg = curva.to_periodogram(maximum_frequency=max_freq,
                                  oversample_factor=oversample)

        freqs = 1 / pg.period.value
        powers = pg.power.value
        ordenado = np.argsort(freqs)
        freqs, powers = freqs[ordenado], powers[ordenado]

        harmonicos = identificar_harmonicos(
            freqs, powers, freq_base, limiar_percentil, tolerancia, max_harm
        )

        if len(harmonicos) == len(harmonicos_previos):
            break
        harmonicos_previos = harmonicos
        max_harm += harm_step

    return harmonicos_previos, max_freq, max_harm - harm_step, pg, freqs, powers


# %% Funções de plotagem

def plotar_curva_luz(curva, tic_id, salvar=False, pasta='plots'):
    """
    Plota a curva de luz bruta (fluxo vs tempo).
    """
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(curva.time.value, curva.flux.value, 'k.', markersize=1)
    plt.title(f"Curva de Luz - TIC {tic_id}")
    plt.xlabel("Tempo (dias)")
    plt.ylabel("Fluxo Normalizado (ppm)")
    plt.grid(True)

    if salvar:
        caminho = os.path.join(pasta, 'curvas_de_luz', f'TIC_{tic_id}_curva.png')
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        plt.savefig(caminho, bbox_inches='tight')
        print(f"📊 Curva de luz salva em: {caminho}")
    plt.show()


def plotar_periodograma(freqs, powers, freq_base, freq_dominante, harmonicos,
                        tic_id, salvar=False, pasta='plots', max_harm=10):
    """
    Plota o periodograma com marcação da frequência base, dominante e dos harmônicos detectados.
    """
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(freqs, powers, label=f'TIC {tic_id}')

    plt.axvline(freq_base, color='r', linestyle='--', label='Freq. Real')
    plt.axvline(1 / freq_dominante, color='g', linestyle='-.', label='Freq. Dominante')

    for n in range(2, max_harm + 1):
        plt.axvline(freq_base * n, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    if harmonicos:
        plt.scatter(*zip(*harmonicos), color='orange', marker='x', label='Harmônicos')

    plt.title(f"Periodograma - TIC {tic_id}")
    plt.xlabel("Frequência (1/dia)")
    plt.ylabel("Potência")
    plt.grid(True)
    plt.legend()

    if salvar:
        caminho = os.path.join(pasta, 'periodogramas', f'TIC_{tic_id}_periodograma.png')
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        plt.savefig(caminho, bbox_inches='tight')
        print(f"📊 Periodograma salvo em: {caminho}")
    plt.show()


# %% Função principal

def analisar_exoplaneta(dados, salvar=False, pasta='plots',
                        limiar_cdf=88.1, tolerancia=0.03,
                        max_harm=100, oversample=15):
    """
    Pipeline principal de análise:
    - Baixa curva de luz
    - Gera periodograma
    - Detecta harmônicos
    - Plota curva e espectro
    """
    tic_id = dados['TIC_ID']
    periodo = dados['period']
    setores = dados['sector']

    print(f"\n🔎 Analisando TIC {tic_id} - Período: {periodo:.5f} dias")

    curva = baixar_curva_luz(tic_id, setores)
    if curva is None:
        print(f"⚠️ Nenhuma curva encontrada para TIC {tic_id} nos setores {setores}")
        return

    plotar_curva_luz(curva, tic_id, salvar, pasta)

    freq_base = 1 / periodo

    harmonicos, freq_usada, harm_usado, pg, freqs, powers = detectar_harmonicos_com_expansao(
        curva,
        freq_base,
        max_harm_inicial=5,
        harm_step=10,
        harm_limite=max_harm,
        limiar_percentil=limiar_cdf,
        tolerancia=tolerancia,
        oversample=oversample
    )

    freq_dominante = 1 / pg.period_at_max_power.value

    plotar_periodograma(freqs, powers, freq_base, 1 / freq_dominante, harmonicos,
                        tic_id, salvar, pasta, max_harm=harm_usado)

    print(f"✅ Finalizado TIC {tic_id} - Harmônicos detectados: {len(harmonicos)}")
    print(f"🔁 Frequência máxima usada: {freq_usada:.3f}, Harmônicos analisados: {harm_usado}")


# %% Execução por CSV

arquivo = '/graduacao/joshuakipper/Documents/ic/exoplanetas/light-curves/dados_exoplanetas/alvos_kp_testes.csv'
tabela = pd.read_csv(arquivo)

tic_especifico = 9725627         # Exemplo: 123456789 para analisar apenas um TIC específico
salvar_graficos = False       # Mude para True para salvar os gráficos
SETOR_MIN = 1
SETOR_MAX = 10

for _, linha in tabela.iterrows():
    tic_id = int(linha['star_name'])
    if tic_especifico and tic_id != tic_especifico:
        continue

    # Seleciona setores válidos dentro dos limites definidos
    setores_raw = str(linha['Sectors']).split(',')
    setores_validos = []
    for s in setores_raw:
        try:
            setor = int(s.strip())
            if SETOR_MIN <= setor <= SETOR_MAX:
                setores_validos.append(setor)
        except ValueError:
            continue

    if not setores_validos:
        print(f"⚠️ Nenhum setor válido para TIC {tic_id}. Ignorando.")
        continue

    alvo = {
        'TIC_ID': tic_id,
        'period': float(linha['orbital_period(days)']),
        'time_transit': float(linha['transit_duration(hours)']),
        'sector': setores_validos
    }

    analisar_exoplaneta(alvo, salvar=salvar_graficos)
