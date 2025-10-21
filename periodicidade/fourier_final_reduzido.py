# final_min.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def salvar_fig(fig, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    fig.savefig(caminho, bbox_inches="tight")

def baixar_curva_luz(tic_id, setores):
    if isinstance(setores, int):
        setores = [setores]
    curvas = []
    for setor in setores:
        try:
            res = lk.search_lightcurve(f"TIC {tic_id}", cadence="short", mission="TESS", author="SPOC", sector=setor)
            if res and len(res) > 0:
                curva = (res.download_all().stitch().remove_nans().normalize(unit="ppm") - 1)
                curvas.append(curva)
        except Exception:
            pass
    if not curvas:
        return None
    try:
        return curvas[0].append(curvas[1:]) if len(curvas) > 1 else curvas[0]
    except Exception:
        return None

def identificar_harmonicos(freqs, powers, f0, pctl, tol, max_harm=50):
    if freqs.size == 0 or powers.size == 0:
        return []
    limiar = np.percentile(powers, pctl)
    idx, _ = find_peaks(powers, height=limiar)
    if idx.size == 0:
        return []
    picos_f, picos_p = freqs[idx], powers[idx]
    harm = []
    for n in range(1, max_harm + 1):
        alvo = n * f0
        m = (picos_f > alvo * (1 - tol)) & (picos_f < alvo * (1 + tol))
        if np.any(m):
            j = np.argmax(picos_p[m])
            harm.append((picos_f[m][j], picos_p[m][j]))
    return harm

def detectar_harmonicos_com_expansao(curva, f0, max_harm_ini=5, step=10, limite=100, pctl=90.0, tol=0.03, oversample=15):
    max_h, prev = max_harm_ini, []
    pg_final = None
    freqs = powers = np.array([])
    max_freq_final = max_harm_ini * f0
    while max_h <= limite:
        max_freq = max_h * f0
        try:
            pg = curva.to_periodogram(maximum_frequency=max_freq, oversample_factor=oversample)
        except Exception:
            break
        freqs = 1.0 / pg.period.value
        powers = pg.power.value
        o = np.argsort(freqs)
        freqs, powers = freqs[o], powers[o]
        harm = identificar_harmonicos(freqs, powers, f0, pctl, tol, max_h)
        if len(harm) == len(prev):
            pg_final, max_freq_final = pg, max_freq
            break
        prev = harm
        max_h += step
    if pg_final is None:
        pg_final, max_freq_final = pg, max_freq
    return prev, max_freq_final, max_h - step, pg_final, freqs, powers

def avaliar_fourier(t, params, f0):
    a0 = params[0]
    coef = params[1:]
    N = len(coef) // 2
    y = np.full_like(t, a0, dtype=float)
    if N == 0:
        return y
    n = np.arange(1, N + 1)
    an, bn = coef[0::2], coef[1::2]
    ang = 2.0 * np.pi * (n[:, None] * f0) * t[None, :]
    y += (an[:, None] * np.cos(ang) + bn[:, None] * np.sin(ang)).sum(axis=0)
    return y

def ajustar_fourier(curva, f0, num_harm, maxfev=20000):
    t, y = curva.time.value, curva.flux.value
    p0 = np.zeros(1 + 2 * num_harm, dtype=float)
    p0[0] = float(np.mean(y))
    def modelo(t_in, a0, *coef):
        return avaliar_fourier(t_in, np.array([a0, *coef], dtype=float), f0)
    try:
        popt, _ = curve_fit(modelo, t, y, p0=p0, maxfev=maxfev)
    except Exception:
        popt = p0
    return popt

def plotar_curva_luz(curva, tic_id, salvar=False, pasta="plots"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curva.time.value, curva.flux.value, ".", ms=1)
    ax.set(title=f"Curva de Luz - TIC {tic_id}", xlabel="Tempo (dias)", ylabel="Fluxo (ppm)")
    if salvar:
        salvar_fig(fig, os.path.join(pasta, "curvas_de_luz", f"TIC_{tic_id}_curva.png"))
    plt.show()

def plotar_periodograma(freqs, powers, f0, period_dom, harmonicos, tic_id, salvar=False, pasta="plots", max_harm=10):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, powers)
    ax.axvline(f0, color="r", linestyle="--", label="f0")
    ax.axvline(1.0 / period_dom, color="g", linestyle="-.", label="f_dom")
    for n in range(2, max_harm + 1):
        ax.axvline(f0 * n, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    if harmonicos:
        ax.scatter(*zip(*harmonicos), color="orange", marker="x", label="harm")
    ax.set(title=f"Periodograma - TIC {tic_id}", xlabel="Frequência (1/dia)", ylabel="Potência")
    ax.legend()
    if salvar:
        salvar_fig(fig, os.path.join(pasta, "periodogramas", f"TIC_{tic_id}_periodograma.png"))
    plt.show()

def calcular_residuos(curva, params, f0):
    t = curva.time.value
    y = curva.flux.value
    y_fit = avaliar_fourier(t, params, f0)
    return t, y - y_fit, y_fit

def periodograma_residuo(t, resid, max_frequency, oversample=15):
    lc_res = lk.LightCurve(time=t, flux=resid)
    pg = lc_res.to_periodogram(maximum_frequency=max_frequency, oversample_factor=oversample)
    freqs = 1.0 / pg.period.value
    powers = pg.power.value
    o = np.argsort(freqs)
    return freqs[o], powers[o], pg

def plotar_periodograma_residuo(freqs, powers, f0=None, tic_id=None, salvar=False, pasta="plots"):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, powers)
    if f0 is not None and len(freqs) > 0:
        nmax = int(np.max(freqs) // f0)
        for n in range(1, max(nmax, 1) + 1):
            ax.axvline(n * f0, color="gray", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.set(title=f"Periodograma do resíduo - TIC {tic_id}" if tic_id else "Periodograma do resíduo",
           xlabel="Frequência (1/dia)", ylabel="Potência")
    if salvar and tic_id is not None:
        salvar_fig(fig, os.path.join(pasta, "residuos", f"TIC_{tic_id}_periodograma_residuo.png"))
    plt.show()

def fase_centrada(t, f0, t0):
    return ((t - t0) * f0 + 0.5) % 1.0 - 0.5

def estimar_t0_pelo_minimo_do_modelo(t, y_fit):
    return float(t[np.argmin(y_fit)])

def plotar_dobramento_centralizado(curva, params, f0, i, t0=None, tic_id=None, salvar=False, pasta="plots", n_model=2000):
    t, y = curva.time.value, curva.flux.value
    y_fit_obs = avaliar_fourier(t, params, f0)
    if t0 is None:
        t0 = estimar_t0_pelo_minimo_do_modelo(t, y_fit_obs)
    fase_dados = fase_centrada(t, f0, t0)
    fase_grid = np.linspace(-0.5, 0.5, n_model)
    t_model = t0 + fase_grid / f0
    y_model = avaliar_fourier(t_model, params, f0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fase_dados, y, ".", ms=1, alpha=0.35, label=f"Dados(setor:{i})")
    ax.plot(fase_grid, y_model, "-", lw=2, alpha=0.95, label="Modelos")
    ax.set(xlabel="Fase", ylabel="Fluxo",
           title=f"Trânsito Dobrado - TIC {tic_id}" if tic_id else "Trânsito Dobrado")
    ax.legend()
    if salvar and tic_id is not None:
        salvar_fig(fig, os.path.join(pasta, "dobramento", f"TIC_{tic_id}_dobramento_centralizado.png"))
    plt.show()
    return t0

def analisar_exoplaneta(dados, i, salvar=False, pasta="plots", pctl=88.1, tol=0.03, max_harm=100, oversample=15, t0_fold=None):
    tic_id = int(dados["TIC_ID"])
    periodo = float(dados["period"])
    setores = list(dados["sector"])
    curva = baixar_curva_luz(tic_id, setores)
    if curva is None:
        return
    #plotar_curva_luz(curva, tic_id, salvar=salvar, pasta=pasta)
    f0 = 1.0 / periodo
    harmonicos, fmax, n_harm, pg, freqs, powers = detectar_harmonicos_com_expansao(
        curva=curva, f0=f0, max_harm_ini=5, step=10, limite=max_harm, pctl=pctl, tol=tol, oversample=oversample
    )
    #period_dom = pg.period_at_max_power.value
    #plotar_periodograma(freqs, powers, f0, period_dom, harmonicos, tic_id, salvar=salvar, pasta=pasta, max_harm=n_harm)
    params = ajustar_fourier(curva, f0=f0, num_harm=max(len(harmonicos), 1))
    #fig, ax = plt.subplots(figsize=(10, 5))
    #t, y = curva.time.value, curva.flux.value
    #y_fit = avaliar_fourier(t, params, f0)
    #ax.plot(t, y, ".", ms=1, label="Dados")
    #ax.plot(t, y_fit, "-", lw=2, label="Ajuste")
    #ax.set(title=f"Ajuste Série de Fourier - TIC {tic_id}", xlabel="Tempo (dias)", ylabel="Fluxo")
    #ax.legend()
    #if salvar:
    #    salvar_fig(fig, os.path.join(pasta, "ajustes_fourier", f"TIC_{tic_id}_ajuste.png"))
    #plt.show()
    #t, resid, _ = calcular_residuos(curva, params, f0)
    #fig, ax = plt.subplots(figsize=(10, 4))
    #ax.plot(t, resid, ".", ms=1)
    #ax.axhline(0.0, color="gray", lw=1)
    #ax.set(title=f"Resíduo temporal - TIC {tic_id}", xlabel="Tempo (dias)", ylabel="Resíduo")
    #if salvar:
    #    salvar_fig(fig, os.path.join(pasta, "residuos", f"TIC_{tic_id}_residuo_tempo.png"))
    #plt.show()
    #freqs_r, powers_r, _ = periodograma_residuo(t, resid, max_frequency=fmax, oversample=oversample)
    #plotar_periodograma_residuo(freqs_r, powers_r, f0=f0, tic_id=tic_id, salvar=salvar, pasta=pasta)
    plotar_dobramento_centralizado(curva, params, f0, i, t0=t0_fold, tic_id=tic_id, salvar=salvar, pasta=pasta, n_model=2000)

for i in [1,2,3,4,5,6,7,8,9,10]:
    if __name__ == "__main__":
    #    ARQUIVO_CSV = "/graduacao/joshuakipper/Documents/ic/exoplanetas/light-curves/dados_exoplanetas/alvos_kp_testes.csv"
        ARQUIVO_CSV = "/home/joshua/Documentos/ufrgs/light-curves-main/dados_exoplanetas/data_ExoFOP(taina).csv"
        TIC_ESPECIFICO = 38846515  # None => todos
        SALVAR_GRAFICOS = False
        SETOR_MIN, SETOR_MAX = i, i
    
        try:
            tabela = pd.read_csv(ARQUIVO_CSV)
        except Exception:
            raise SystemExit(1)
    
        for _, linha in tabela.iterrows():
            try:
                tic_id = int(linha["star_name"])
            except Exception:
                continue
            if TIC_ESPECIFICO is not None and tic_id != TIC_ESPECIFICO:
                continue
            setores_validos = []
            for s in str(linha.get("Sectors", "")).split(","):
                try:
                    setor = int(s.strip())
                    if SETOR_MIN <= setor <= SETOR_MAX:
                        setores_validos.append(setor)
                except ValueError:
                    pass
            if not setores_validos:
                continue
            try:
                alvo = {
                    "TIC_ID": tic_id,
                    "period": float(linha["orbital_period(days)"]),
                    "sector": setores_validos
                }
            except Exception:
                continue
            analisar_exoplaneta(
                dados=alvo,
                i=i,
                salvar=SALVAR_GRAFICOS,
                pasta="plots",
                pctl=88.1,
                tol=0.03,
                max_harm=100,
                oversample=15,
                t0_fold=None
            )
