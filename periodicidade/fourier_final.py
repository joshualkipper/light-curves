# %%
"""
===========================================
Analisador de Curvas de Luz (TESS) — Fourier
===========================================

Descrição geral
---------------
Este script automatiza a análise de curvas de luz de alvos do TESS:
1) Baixa e prepara os dados com Lightkurve.
2) Calcula o periodograma e detecta harmônicos (n · f0).
3) Ajusta uma série de Fourier (truncada) por *curve_fit*.
4) Calcula e inspeciona os resíduos (tempo + periodograma).
5) Realiza o dobramento (phase-folding) **centralizado em φ=0**,
   exibindo **apenas os dados e o modelo**.

Dependências
------------
- numpy, pandas, matplotlib
- scipy (signal, optimize)
- lightkurve

Como usar (passo a passo)
-------------------------
- Ajuste o caminho `ARQUIVO_CSV` no bloco final (execução em lote).
- (Opcional) Defina `TIC_ESPECIFICO` para rodar apenas um alvo.
- Execute célula a célula no Spyder.

Estrutura do código
-------------------
- Configuração básica (gráficos e logging leve)
- Utilidades (salvar figuras)
- Download e preparo da curva (Lightkurve)
- Periodograma e detecção de harmônicos
- Série de Fourier: avaliação e ajuste
- Gráficos (curva, periodograma)
- Resíduos e seu periodograma
- Dobramento centralizado (apenas dados + modelo)
- Pipeline principal
- Execução em lote (via CSV)
"""

# %%
"""
[Seção] Configuração básica
Objetivo: Ajustar estilo visual dos gráficos e configurar um logging simples.
Entradas: —
Saídas : —
Obs.   : Mantido o mínimo necessário para clareza; sem estilos avançados.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightkurve as lk
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# Logging simples (console)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Estilo gráfico básico
plt.rcParams.update({
    "figure.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 1.5
})

# %%
"""
[Seção] Utilidades (I/O de figuras)
Objetivo: Salvar figuras garantindo a criação do diretório.
Entradas: fig (matplotlib.figure.Figure), caminho (str)
Saídas : arquivo de imagem no disco
Obs.   : Uso interno pelas funções de plotagem.
"""
def salvar_fig(fig, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    fig.savefig(caminho, bbox_inches="tight")
    logging.info(f"Figura salva: {caminho}")

# %%
"""
[Seção] Download e preparo de dados (Lightkurve)
Objetivo: Baixar curva(s) de luz TESS por TIC e setores; normalizar e limpar.
Entradas: tic_id (int), setores (lista de ints)
Saídas : lightkurve.LightCurve (ou None em falha)
Obs.   : Combina múltiplos setores via stitch e normaliza para ppm.
"""
def baixar_curva_luz(tic_id, setores):
    if isinstance(setores, int):
        setores = [setores]

    curvas = []
    for setor in setores:
        try:
            resultado = lk.search_lightcurve(
                f"TIC {tic_id}", cadence="short", mission="TESS", author="SPOC", sector=setor
            )
        except Exception as e:
            logging.warning(f"Busca falhou (TIC {tic_id}, setor {setor}): {e}")
            continue

        if resultado is None or len(resultado) == 0:
            logging.warning(f"Sem produto (TIC {tic_id}, setor {setor}).")
            continue

        try:
            curva = (
                resultado.download_all()
                .stitch()
                .remove_nans()
                .normalize(unit="ppm") - 1
            )
            curvas.append(curva)
        except Exception as e:
            logging.warning(f"Download/processamento falhou (TIC {tic_id}, setor {setor}): {e}")

    if not curvas:
        return None

    try:
        return curvas[0].append(curvas[1:]) if len(curvas) > 1 else curvas[0]
    except Exception as e:
        logging.error(f"Falha ao concatenar setores (TIC {tic_id}): {e}")
        return None

# %%
"""
[Seção] Periodograma e detecção de harmônicos
Objetivo: Construir periodograma até estabilizar nº de harmônicos; identificar picos n·f0.
Entradas: curva (LightCurve), freq_base (float), parâmetros de busca
Saídas : harmonicos (lista[(freq, pot)]), max_freq, harm_uso, pg, freqs, powers
Obs.   : Critério de parada simples: estabilização do total de harmônicos.
"""
def identificar_harmonicos(freqs, powers, freq_base, limiar_percentil, tolerancia, max_harm=50):
    if freqs.size == 0 or powers.size == 0:
        return []

    limiar = np.percentile(powers, limiar_percentil)
    idx, _ = find_peaks(powers, height=limiar)
    if idx.size == 0:
        return []

    picos_f = freqs[idx]
    picos_p = powers[idx]
    harmonicos = []

    for n in range(1, max_harm + 1):
        alvo = n * freq_base
        faixa = (picos_f > alvo * (1 - tolerancia)) & (picos_f < alvo * (1 + tolerancia))
        if not np.any(faixa):
            continue
        loc = np.argmax(picos_p[faixa])
        harmonicos.append((picos_f[faixa][loc], picos_p[faixa][loc]))

    return harmonicos


def detectar_harmonicos_com_expansao(
    curva,
    freq_base,
    max_harm_inicial=5,
    harm_step=10,
    harm_limite=100,
    limiar_percentil=90.0,
    tolerancia=0.03,
    oversample=15
):
    max_harm = max_harm_inicial
    harmonicos_previos = []
    pg_final, freqs_final, powers_final = None, np.array([]), np.array([])
    max_freq_final = max_harm_inicial * freq_base

    while max_harm <= harm_limite:
        max_freq = max_harm * freq_base
        try:
            pg = curva.to_periodogram(maximum_frequency=max_freq, oversample_factor=oversample)
        except Exception as e:
            logging.error(f"Periodograma falhou: {e}")
            break

        freqs = 1.0 / pg.period.value
        powers = pg.power.value
        ordem = np.argsort(freqs)
        freqs, powers = freqs[ordem], powers[ordem]

        harmonicos = identificar_harmonicos(freqs, powers, freq_base, limiar_percentil, tolerancia, max_harm)

        if len(harmonicos) == len(harmonicos_previos):
            pg_final, freqs_final, powers_final = pg, freqs, powers
            max_freq_final = max_freq
            break

        harmonicos_previos = harmonicos
        max_harm += harm_step

    harm_uso = max_harm - harm_step
    if pg_final is None:
        pg_final, freqs_final, powers_final, max_freq_final = pg, freqs, powers, max_freq

    return harmonicos_previos, max_freq_final, harm_uso, pg_final, freqs_final, powers_final

# %%
"""
[Seção] Série de Fourier — avaliação e ajuste
Objetivo: Avaliar f(t) e ajustar coeficientes por curve_fit.
Entradas: t (array), params (a0, a1, b1, ..., aN, bN), f0 (freq. base)
Saídas : avaliar_fourier -> y(t) ; ajustar_fourier -> params otimizados
Obs.   : Sem variáveis globais; modelo fechado simples para o fit.
"""
def avaliar_fourier(t, params, f0):
    a0 = params[0]
    coef = params[1:]
    N = len(coef) // 2
    y = np.full_like(t, a0, dtype=float)
    if N == 0:
        return y
    n = np.arange(1, N + 1)
    an = coef[0::2]
    bn = coef[1::2]
    ang = 2.0 * np.pi * (n[:, None] * f0) * t[None, :]
    y += (an[:, None] * np.cos(ang) + bn[:, None] * np.sin(ang)).sum(axis=0)
    return y


def ajustar_fourier(curva, freq_base, num_harm, maxfev=20000):
    t = curva.time.value
    y = curva.flux.value

    # Chute inicial: a0 = média(y); demais = 0
    p0 = np.zeros(1 + 2 * num_harm, dtype=float)
    p0[0] = float(np.mean(y))

    def modelo(t_in, a0, *coef):
        params = np.array([a0, *coef], dtype=float)
        return avaliar_fourier(t_in, params, freq_base)

    try:
        popt, _ = curve_fit(modelo, t, y, p0=p0, maxfev=maxfev)
    except Exception as e:
        logging.error(f"Ajuste de Fourier falhou: {e}")
        popt = p0  # fallback simples

    return popt

# %%
"""
[Seção] Gráficos principais (curva e periodograma)
Objetivo: Visualizar curva bruta e periodograma com marcações.
Entradas: arrays e metadados (freq_base, período dominante, etc.)
Saídas : figuras interativas
Obs.   : Sem marcação de picos no resíduo; apenas guias de harmônicos.
"""
def plotar_curva_luz(curva, tic_id, salvar=False, pasta="plots"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curva.time.value, curva.flux.value, "k.", markersize=1)
    ax.set(title=f"Curva de Luz - TIC {tic_id}", xlabel="Tempo (dias)", ylabel="Fluxo normalizado (ppm)")
    if salvar:
        salvar_fig(fig, os.path.join(pasta, "curvas_de_luz", f"TIC_{tic_id}_curva.png"))
    plt.show()


def plotar_periodograma(freqs, powers, freq_base, period_dominante, harmonicos, tic_id,
                        salvar=False, pasta="plots", max_harm=10):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, powers, label=f"TIC {tic_id}")

    ax.axvline(freq_base, color="r", linestyle="--", label="Freq. (cat.)")
    ax.axvline(1.0 / period_dominante, color="g", linestyle="-.", label="Freq. dominante")
    for n in range(2, max_harm + 1):
        ax.axvline(freq_base * n, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    if harmonicos:
        ax.scatter(*zip(*harmonicos), color="orange", marker="x", label="Harmônicos")

    ax.set(title=f"Periodograma - TIC {tic_id}", xlabel="Frequência (1/dia)", ylabel="Potência")
    ax.legend()

    if salvar:
        salvar_fig(fig, os.path.join(pasta, "periodogramas", f"TIC_{tic_id}_periodograma.png"))
    plt.show()

# %%
"""
[Seção] Resíduos (tempo + periodograma)
Objetivo: Calcular resíduo y - y_fit e seu periodograma.
Entradas: curva (LightCurve), params Fourier, freq_base, max_frequency
Saídas : (t, resid, y_fit), e gráfico do periodograma de resíduo
Obs.   : Periodograma do resíduo com guias nos múltiplos de f0.
"""
def calcular_residuos(curva, params, freq_base):
    t = curva.time.value
    y = curva.flux.value
    y_fit = avaliar_fourier(t, params, freq_base)
    resid = y - y_fit
    return t, resid, y_fit


def periodograma_residuo(t, resid, max_frequency, oversample=15):
    lc_res = lk.LightCurve(time=t, flux=resid)
    pg = lc_res.to_periodogram(maximum_frequency=max_frequency, oversample_factor=oversample)
    freqs = 1.0 / pg.period.value
    powers = pg.power.value
    ordem = np.argsort(freqs)
    return freqs[ordem], powers[ordem], pg


def plotar_periodograma_residuo(freqs, powers, freq_base=None, tic_id=None, salvar=False, pasta="plots"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, powers, label="Resíduo")

    if freq_base is not None and len(freqs) > 0:
        nmax = int(np.max(freqs) // freq_base)
        for n in range(1, max(nmax, 1) + 1):
            ax.axvline(n * freq_base, color="gray", linestyle="--", alpha=0.35, linewidth=0.8)

    ax.set(title=f"Periodograma do resíduo - TIC {tic_id}" if tic_id else "Periodograma do resíduo",
           xlabel="Frequência (1/dia)", ylabel="Potência")
    ax.legend()

    if salvar and tic_id is not None:
        salvar_fig(fig, os.path.join(pasta, "residuos", f"TIC_{tic_id}_periodograma_residuo.png"))
    plt.show()

# %%
"""
[Seção] Dobramento (apenas dados + modelo), centralizado
Objetivo: Dobrar a curva na fase φ ∈ [-0.5, 0.5], centralizando o trânsito em φ=0.
Entradas: curva (LightCurve), params Fourier, freq_base, t0 (opcional)
Saídas : gráfico; retorna t0 utilizado
Obs.   : Modelo suave é gerado numa grade de fase; dados são só dispersão.
"""
def fase_centrada(t, f0, t0):
    return ((t - t0) * f0 + 0.5) % 1.0 - 0.5

def estimar_t0_pelo_minimo_do_modelo(t, y_fit):
    return float(t[np.argmin(y_fit)])

def plotar_dobramento_centralizado(curva, params, freq_base, t0=None,
                                   tic_id=None, salvar=False, pasta="plots", n_model=2000):
    t = curva.time.value
    y = curva.flux.value

    # Modelo nos tempos observados (para estimar t0 se necessário)
    y_fit_obs = avaliar_fourier(t, params, freq_base)
    if t0 is None:
        t0 = estimar_t0_pelo_minimo_do_modelo(t, y_fit_obs)

    # Fase dos dados (centralizada)
    fase_dados = fase_centrada(t, freq_base, t0)

    # Modelo suave (uma volta)
    fase_grid = np.linspace(-0.5, 0.5, n_model)
    t_model = t0 + fase_grid / freq_base
    y_model = avaliar_fourier(t_model, params, freq_base)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fase_dados, y, ".", ms=1, alpha=0.35, label="Dados (fase)")
    ax.plot(fase_grid, y_model, "-", lw=2, alpha=0.95, label="Modelo Fourier")
    ax.set(xlim=(-0.5, 0.5), xlabel="Fase (ciclos, centrado em 0)", ylabel="Fluxo normalizado (ppm)",
           title=f"Dobramento (centralizado) - TIC {tic_id}" if tic_id else "Dobramento (centralizado)")
    ax.legend()

    if salvar and tic_id is not None:
        salvar_fig(fig, os.path.join(pasta, "dobramento", f"TIC_{tic_id}_dobramento_centralizado.png"))
    plt.show()
    return t0

# %%
"""
[Seção] Pipeline principal
Objetivo: Orquestrar todo o fluxo para um alvo.
Entradas: dict com {'TIC_ID', 'period', 'sector', ...}; opções de execução e plot
Saídas : gráficos e mensagens no console
Obs.   : Mantém mensagens curtas e claras sobre o andamento.
"""
def analisar_exoplaneta(dados, salvar=False, pasta="plots",
                        limiar_percentil=88.1, tolerancia=0.03,
                        max_harm=100, oversample=15, t0_fold=None):
    tic_id = int(dados["TIC_ID"])
    periodo = float(dados["period"])
    setores = list(dados["sector"])

    logging.info(f"TIC {tic_id} | Período (catálogo) = {periodo:.6f} d")

    curva = baixar_curva_luz(tic_id, setores)
    if curva is None:
        logging.error(f"Nenhuma curva encontrada (TIC {tic_id}, setores {setores}).")
        return

    # 1) Curva bruta
    plotar_curva_luz(curva, tic_id, salvar=salvar, pasta=pasta)

    # 2) Frequência base
    freq_base = 1.0 / periodo

    # 3) Harmônicos com expansão
    harmonicos, freq_usada, harm_usado, pg, freqs, powers = detectar_harmonicos_com_expansao(
        curva=curva,
        freq_base=freq_base,
        max_harm_inicial=5,
        harm_step=10,
        harm_limite=max_harm,
        limiar_percentil=limiar_percentil,
        tolerancia=tolerancia,
        oversample=oversample
    )
    period_dominante = pg.period_at_max_power.value

    # 4) Periodograma
    plotar_periodograma(freqs, powers, freq_base, period_dominante, harmonicos, tic_id,
                        salvar=salvar, pasta=pasta, max_harm=harm_usado)

    # 5) Ajuste Fourier
    num_harm = max(len(harmonicos), 1)
    params = ajustar_fourier(curva, freq_base=freq_base, num_harm=num_harm)

    # 6) Dados vs ajuste (tempo)
    fig, ax = plt.subplots(figsize=(10, 5))
    t = curva.time.value
    y = curva.flux.value
    y_fit = avaliar_fourier(t, params, freq_base)
    ax.plot(t, y, "k.", ms=1, label="Dados")
    ax.plot(t, y_fit, "r-", lw=2, label="Ajuste Fourier")
    ax.set(title=f"Ajuste Série de Fourier - TIC {tic_id}", xlabel="Tempo (dias)", ylabel="Fluxo normalizado (ppm)")
    ax.legend()
    if salvar:
        salvar_fig(fig, os.path.join(pasta, "ajustes_fourier", f"TIC_{tic_id}_ajuste.png"))
    plt.show()

    # 7) Resíduos + periodograma
    t, resid, y_fit = calcular_residuos(curva, params, freq_base)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, resid, "k.", ms=1, label="Resíduo (y - y_fit)")
    ax.axhline(0.0, color="gray", lw=1)
    ax.set(title=f"Resíduo temporal - TIC {tic_id}", xlabel="Tempo (dias)", ylabel="Resíduo (ppm)")
    ax.legend()
    if salvar:
        salvar_fig(fig, os.path.join(pasta, "residuos", f"TIC_{tic_id}_residuo_tempo.png"))
    plt.show()

    freqs_r, powers_r, _ = periodograma_residuo(t, resid, max_frequency=freq_usada, oversample=oversample)
    plotar_periodograma_residuo(freqs_r, powers_r, freq_base=freq_base, tic_id=tic_id, salvar=salvar, pasta=pasta)

    # 8) Dobramento centralizado (apenas dados + modelo)
    t0_usado = plotar_dobramento_centralizado(curva, params, freq_base, t0=t0_fold,
                                              tic_id=tic_id, salvar=salvar, pasta=pasta, n_model=2000)
    logging.info(f"Dobramento concluído (t0 = {t0_usado:.6f} d).")
    logging.info(f"Finalizado TIC {tic_id} | Harmônicos: {len(harmonicos)} | f_max: {freq_usada:.3f} | n_harm: {harm_usado}")

# %%
"""
[Seção] Execução em lote (via CSV)
Objetivo: Ler um CSV de alvos e executar o pipeline por linha.
Entradas: ARQUIVO_CSV, filtros e opções no bloco abaixo
Saídas : gráficos em tela (e em disco se 'salvar=True')
Obs.   : Ajuste SETOR_MIN/MAX conforme o recorte desejado.
"""
if __name__ == "__main__":
    ARQUIVO_CSV = "/graduacao/joshuakipper/Documents/ic/exoplanetas/light-curves/dados_exoplanetas/alvos_kp_testes.csv"
    TIC_ESPECIFICO = 25155310 # ex.: 9725627 ; None => todos
    SALVAR_GRAFICOS = False
    SETOR_MIN, SETOR_MAX = 1, 10

    try:
        tabela = pd.read_csv(ARQUIVO_CSV)
    except Exception as e:
        logging.error(f"Falha ao ler CSV: {e}")
        raise SystemExit(1)

    for _, linha in tabela.iterrows():
        # TIC
        try:
            tic_id = int(linha["star_name"])
        except Exception:
            logging.warning("Linha sem 'star_name' válido; ignorando.")
            continue
        if TIC_ESPECIFICO is not None and tic_id != TIC_ESPECIFICO:
            continue

        # Setores válidos
        setores_validos = []
        for s in str(linha.get("Sectors", "")).split(","):
            try:
                setor = int(s.strip())
                if SETOR_MIN <= setor <= SETOR_MAX:
                    setores_validos.append(setor)
            except ValueError:
                continue
        if not setores_validos:
            logging.warning(f"Nenhum setor válido para TIC {tic_id}; ignorando.")
            continue

        # Alvo
        try:
            alvo = {
                "TIC_ID": tic_id,
                "period": float(linha["orbital_period(days)"]),
                "time_transit": float(linha.get("transit_duration(hours)", np.nan)),
                "sector": setores_validos
            }
        except Exception as e:
            logging.warning(f"Linha malformada para TIC {tic_id}: {e}")
            continue

        analisar_exoplaneta(
            dados=alvo,
            salvar=SALVAR_GRAFICOS,
            pasta="plots",
            limiar_percentil=88.1,
            tolerancia=0.03,
            max_harm=100,
            oversample=15,
            t0_fold=None  # Defina um t0 conhecido aqui, se quiser ancorar a fase
        )
