# %% [markdown]
# Pipeline de Trânsitos com Série de Fourier
#
# Pipeline modular para análise de trânsitos planetários (TESS/SPOC) com
# ajuste por série de Fourier, identificação de harmônicos em periodograma
# e geração de gráficos configuráveis.
#
# Pontos chave
# - Seleção de gráficos: qualquer subconjunto entre
#   {"transits_full", "transit_folded", "periodogram", "full_with_model", "folded_with_model"}.
# - Setores: modos "combined" (única análise), "separate" (uma por setor) ou "both";
#   com opção de sobreposição ("overlay") para visual.
# - Salvamento determinístico: respeito estrito a `save` e `show_plots`.
# - Logs de debug com `verbose=True`.
# - Organização por TIC: <outdir>/TIC_<id>/ com subpastas por modo/setor.

# %%
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Dict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# %% [markdown]
# Configuração principal

# %%
# Modo de composição por setor
# - "combined": analisa setores juntos (uma única análise)
# - "separate": roda uma análise por setor (figuras separadas)
# - "both": executa ambos os modos

PLOT_KEYS = (
    "transits_full",          # (1) trânsitos completos (tempo x fluxo)
    "transit_folded",         # (2) trânsito dobrado (fase x fluxo)
    "periodogram",            # (3) periodograma (freq x potência)
    "full_with_model",        # (4) tempo x (dados + modelo de Fourier)
    "folded_with_model",      # (5) fase x (dados + modelo de Fourier)
)


@dataclass
class Config:
    """
    Configurações globais do pipeline.

    Parâmetros
    ----------
    csv_path : str ou None
        Caminho para o CSV de entrada (ExoFOP).
    tic_only : int ou None
        Se definido, filtra a análise para um único TIC.
    sectors : iterável de int ou None
        Lista de setores a analisar; None usa os do CSV.
    sector_mode : {"combined", "separate", "both"}
        "combined": única análise; "separate": por setor; "both": ambos.
    sector_overlay : bool
        No modo "combined", se True plota sobreposição (visual).
    which_plots : iterável de chaves em PLOT_KEYS
        Subconjunto de gráficos a gerar.
    save : bool
        Se True, salva figuras em `outdir`.
    outdir : str
        Diretório base para salvar figuras.
    fig_dpi : int
        DPI das figuras salvas.
    pctl : float
        Percentil para limiar de picos no periodograma.
    tol : float
        Tolerância relativa para casar n*f0.
    max_harm_limit : int
        Limite superior de harmônicos na expansão da frequência máxima.
    oversample : int
        Fator de superamostragem do periodograma.
    t0_fold : float ou None
        Época de referência para dobramento; None estima automaticamente.
    show_plots : bool
        Se True, exibe figuras; se False, fecha após salvar.
    verbose : bool
        Logs simples (info/salvo/erros).
    """

    # Seleção de alvos/entrada
    csv_path: Optional[str] = None
    tic_only: Optional[int] = None

    # Setores a analisar
    sectors: Optional[Iterable[int]] = None
    sector_mode: str = "combined"        # "combined" | "separate" | "both"
    sector_overlay: bool = True          # apenas para o modo combined

    # Gráficos desejados
    which_plots: Iterable[str] = field(default_factory=lambda: PLOT_KEYS)

    # Saída de figuras
    save: bool = False
    outdir: str = "plots"
    fig_dpi: int = 150

    # Análise harmônicos / periodograma / ajuste
    pctl: float = 88.1
    tol: float = 0.03
    max_harm_limit: int = 100
    oversample: int = 15

    # Dobramento
    t0_fold: Optional[float] = None

    # Controle de exibição e logs
    show_plots: bool = True
    verbose: bool = True

# %% [markdown]
# Utilidades de E/S e figuras

# %%
def _ensure_dir(path: str) -> None:
    """
    Garante que o diretório de `path` exista.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _save_fig(fig: plt.Figure, path: str, cfg: Config) -> None:
    """
    Salva a figura conforme `cfg.save` e imprime log se `verbose=True`.
    """
    if not cfg.save:
        return
    try:
        _ensure_dir(path)
        fig.savefig(path, dpi=cfg.fig_dpi, bbox_inches="tight")
        if cfg.verbose:
            print(f"[SALVO] {path}")
    except Exception as e:
        print(f"[ERRO ao salvar] {path} -> {e}")


def _finalizar_fig(fig: plt.Figure, cfg: Config, save_path: str | None = None) -> None:
    """
    Centraliza salvar/mostrar/fechar:
    - Salva se `save_path` for dado e `cfg.save=True`.
    - Respeita `cfg.show_plots` (plt.ion()/plt.ioff() setado no início da execução).
    - Fecha a figura quando `show_plots=False` para evitar auto-exibição do Spyder.
    """
    if save_path is not None:
        _save_fig(fig, save_path, cfg)
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig)


def _norm_suffix(sfx: Optional[str]) -> str:
    """
    Normaliza sufixo textual para composição no nome do arquivo.
    """
    if not sfx:
        return ""
    return sfx if sfx.startswith("_") else f"_{sfx}"


def _should(plot_key: str, cfg: Config) -> bool:
    """
    Retorna True se a chave de gráfico deve ser gerada.
    """
    return plot_key in cfg.which_plots


def _tic_dir(tic_id: int, cfg: Config, subdir: Optional[str] = None) -> str:
    """
    Diretório de saída por TIC, com subpasta opcional:
    <outdir>/TIC_<id>[/<subdir>]
    """
    base = os.path.join(cfg.outdir, f"TIC_{tic_id}")
    return os.path.join(base, subdir) if subdir else base

# %% [markdown]
# Download e preparação da curva de luz (Lightkurve)

# %%
def baixar_curvas_por_setor(tic_id: int, setores: Optional[Iterable[int]]) -> Tuple[Optional[lk.LightCurve], List[Tuple[int, lk.LightCurve]]]:
    """
    Baixa curvas por setor e também retorna uma curva unificada (stitch/append).
    Retorna (curva_unica, lista_por_setor). Se setores=None, tenta todos os
    setores disponíveis e só retorna a curva unificada.
    """
    curvas: List[Tuple[int, lk.LightCurve]] = []
    curva_unica: Optional[lk.LightCurve] = None

    try:
        if setores is None:
            # Sem lista explícita: baixa tudo de uma vez e retorna apenas unificada
            res = lk.search_lightcurve(f"TIC {tic_id}", cadence="short", mission="TESS", author="SPOC")
            if not res:
                return None, []
            curva_unica = res.download_all().stitch().remove_nans().normalize(unit="ppm") - 1
            return curva_unica, []
        else:
            setores = sorted(set(int(s) for s in setores))
            lcs = []
            for setor in setores:
                res = lk.search_lightcurve(f"TIC {tic_id}", cadence="short", mission="TESS", author="SPOC", sector=setor)
                if res and len(res) > 0:
                    lc = (res.download_all().stitch().remove_nans().normalize(unit="ppm") - 1)
                    curvas.append((setor, lc))
                    lcs.append(lc)
            if lcs:
                try:
                    curva_unica = lcs[0] if len(lcs) == 1 else lcs[0].append(lcs[1:])
                except Exception:
                    curva_unica = None
            return curva_unica, curvas
    except Exception:
        return None, []

# Shim de compatibilidade: mantém o nome antigo para chamadas herdadas.
def baixar_curva_luz(tic_id: int, setores: Optional[Iterable[int]]):
    """
    Compatibilidade: retorna apenas a curva unificada, ignorando a lista por setor.
    """
    curva_unica, _ = baixar_curvas_por_setor(tic_id, setores)
    return curva_unica

# %% [markdown]
# Identificação de harmônicos e periodograma

# %%
def identificar_harmonicos(
    freqs: np.ndarray,
    powers: np.ndarray,
    f0: float,
    pctl: float,
    tol: float,
    max_harm: int = 50,
) -> List[Tuple[float, float]]:
    """
    Seleciona picos compatíveis com harmônicos de f0.
    """
    if freqs.size == 0 or powers.size == 0:
        return []

    limiar = np.percentile(powers, pctl)
    idx, _ = find_peaks(powers, height=limiar)
    if idx.size == 0:
        return []

    picos_f, picos_p = freqs[idx], powers[idx]
    harm: List[Tuple[float, float]] = []

    for n in range(1, max_harm + 1):
        alvo = n * f0
        m = (picos_f > alvo * (1 - tol)) & (picos_f < alvo * (1 + tol))
        if np.any(m):
            j = np.argmax(picos_p[m])
            harm.append((picos_f[m][j], picos_p[m][j]))

    return harm


def detectar_harmonicos_com_expansao(
    curva: lk.LightCurve,
    f0: float,
    max_harm_ini: int = 5,
    step: int = 10,
    limite: int = 100,
    pctl: float = 90.0,
    tol: float = 0.03,
    oversample: int = 15,
):
    """
    Expande progressivamente a frequência máxima até estabilizar os harmônicos.
    """
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

# %% [markdown]
# Série de Fourier

# %%
def avaliar_fourier(t: np.ndarray, params: np.ndarray, f0: float) -> np.ndarray:
    """
    Avalia a série de Fourier em instantes t.
    """
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


def ajustar_fourier(curva: lk.LightCurve, f0: float, num_harm: int, maxfev: int = 20000) -> np.ndarray:
    """
    Ajusta série de Fourier de ordem num_harm via mínimos quadrados.
    """
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

# %% [markdown]
# Dobramento e épocas de referência

# %%
def fase_centrada(t: np.ndarray, f0: float, t0: float) -> np.ndarray:
    """
    Converte tempos em fase centrada no intervalo [-0.5, 0.5].
    """
    return ((t - t0) * f0 + 0.5) % 1.0 - 0.5


def estimar_t0_pelo_minimo_do_modelo(t: np.ndarray, y_fit: np.ndarray) -> float:
    """
    Estima t0 pelo mínimo de y_fit.
    """
    return float(t[np.argmin(y_fit)])

# %% [markdown]
# Funções de plotagem

# %%
def plot_transits_full(curva: lk.LightCurve, tic_id: int, cfg: Config, filename_suffix: str = "", subdir: Optional[str] = None) -> plt.Figure:
    """Plota curva de luz completa (tempo x fluxo)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curva.time.value, curva.flux.value, ".", ms=1)
    ax.set(title=f"Curva de Luz - TIC {tic_id}", xlabel="Tempo (dias)", ylabel="Fluxo (ppm)")
    sfx = _norm_suffix(filename_suffix)
    caminho = os.path.join(_tic_dir(tic_id, cfg, subdir), f"curva{sfx}.png")
    _finalizar_fig(fig, cfg, save_path=caminho)
    return fig


def plot_transits_full_overlay(curvas_por_setor: List[Tuple[int, lk.LightCurve]], tic_id: int, cfg: Config, subdir: Optional[str] = "overlay") -> plt.Figure:
    """Plota várias curvas (um traço por setor) sobrepostas no mesmo eixo."""
    fig, ax = plt.subplots(figsize=(10, 5))
    setores = []
    for setor, lc in sorted(curvas_por_setor, key=lambda x: x[0]):
        setores.append(int(setor))
        ax.plot(lc.time.value, lc.flux.value, ".", ms=1, alpha=0.7, label=f"setor {setor}")
    ax.set(title=f"Curva de Luz (sobreposição de setores) - TIC {tic_id}", xlabel="Tempo (dias)", ylabel="Fluxo (ppm)")
    ax.legend(ncol=3, fontsize=8)
    tag = "S" + "+S".join(str(s) for s in setores) if setores else "S?"
    caminho = os.path.join(_tic_dir(tic_id, cfg, subdir), f"curva_overlay_{tag}.png")
    _finalizar_fig(fig, cfg, save_path=caminho)
    return fig


def plot_periodogram(
    freqs: np.ndarray,
    powers: np.ndarray,
    f0: float,
    period_dom: Optional[float],
    harmonicos: List[Tuple[float, float]],
    tic_id: int,
    cfg: Config,
    max_harm: int = 10,
    filename_suffix: str = "",
    subdir: Optional[str] = None,
) -> plt.Figure:
    """Plota o periodograma e marca f0, a frequência dominante e guias harmônicas."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, powers)

    ax.axvline(f0, color="r", linestyle="--", label="f0")
    if period_dom is not None and period_dom > 0:
        ax.axvline(1.0 / period_dom, color="g", linestyle="-.", label="f_dom")
    for n in range(2, max_harm + 1):
        ax.axvline(f0 * n, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    if harmonicos:
        ax.scatter(*zip(*harmonicos), color="orange", marker="x", label="harm")

    ax.set(title=f"Periodograma - TIC {tic_id}", xlabel="Frequência (1/dia)", ylabel="Potência")
    ax.legend()
    sfx = _norm_suffix(filename_suffix)
    caminho = os.path.join(_tic_dir(tic_id, cfg, subdir), f"periodograma{sfx}.png")
    _finalizar_fig(fig, cfg, save_path=caminho)
    return fig


def plot_full_with_model(curva: lk.LightCurve, params: np.ndarray, f0: float, tic_id: int, cfg: Config, filename_suffix: str = "", subdir: Optional[str] = None) -> plt.Figure:
    """Plota dados no tempo com o modelo de Fourier sobreposto."""
    t, y = curva.time.value, curva.flux.value
    y_fit = avaliar_fourier(t, params, f0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, y, ".", ms=1, label="Dados")
    ax.plot(t, y_fit, "-", lw=2, label="Ajuste")
    ax.set(title=f"Ajuste Série de Fourier - TIC {tic_id}", xlabel="Tempo (dias)", ylabel="Fluxo")
    ax.legend()
    sfx = _norm_suffix(filename_suffix)
    caminho = os.path.join(_tic_dir(tic_id, cfg, subdir), f"ajuste{sfx}.png")
    _finalizar_fig(fig, cfg, save_path=caminho)
    return fig


def plot_transit_folded(
    curva: lk.LightCurve,
    params: np.ndarray,
    f0: float,
    tic_id: int,
    cfg: Config,
    t0: Optional[float] = None,
    n_model: int = 2000,
    filename_suffix: str = "",
    subdir: Optional[str] = None,
) -> Tuple[plt.Figure, float]:
    """Plota o trânsito dobrado com o modelo em fase."""
    t, y = curva.time.value, curva.flux.value

    y_fit_obs = avaliar_fourier(t, params, f0)
    if t0 is None:
        t0 = estimar_t0_pelo_minimo_do_modelo(t, y_fit_obs)

    fase_dados = fase_centrada(t, f0, t0)
    fase_grid = np.linspace(-0.5, 0.5, n_model)
    t_model = t0 + fase_grid / f0
    y_model = avaliar_fourier(t_model, params, f0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fase_dados, y, ".", ms=1, alpha=0.35, label="Dados")
    ax.plot(fase_grid, y_model, "-", lw=2, alpha=0.95, label="Modelo")
    ax.set(xlabel="Fase", ylabel="Fluxo", title=f"Trânsito Dobrado - TIC {tic_id}")
    ax.legend()
    sfx = _norm_suffix(filename_suffix)
    caminho = os.path.join(_tic_dir(tic_id, cfg, subdir), f"dobramento_centralizado{sfx}.png")
    _finalizar_fig(fig, cfg, save_path=caminho)
    return fig, t0

# %% [markdown]
# Periodograma do resíduo (opcional)

# %%
def calcular_residuos(curva: lk.LightCurve, params: np.ndarray, f0: float):
    """
    Retorna tempos, resíduos e ajuste do modelo.
    """
    t = curva.time.value
    y = curva.flux.value
    y_fit = avaliar_fourier(t, params, f0)
    return t, y - y_fit, y_fit


def periodograma_residuo(t: np.ndarray, resid: np.ndarray, max_frequency: float, oversample: int = 15):
    """
    Computa periodograma do resíduo.
    """
    lc_res = lk.LightCurve(time=t, flux=resid)
    pg = lc_res.to_periodogram(maximum_frequency=max_frequency, oversample_factor=oversample)
    freqs = 1.0 / pg.period.value
    powers = pg.power.value
    o = np.argsort(freqs)
    return freqs[o], powers[o], pg


def plot_periodograma_residuo(
    freqs: np.ndarray,
    powers: np.ndarray,
    f0: Optional[float],
    tic_id: Optional[int],
    cfg: Config,
    filename_suffix: str = "",
    subdir: Optional[str] = None,
) -> plt.Figure:
    """
    Plota periodograma do resíduo e marca n*f0 quando aplicável.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, powers)

    if f0 is not None and len(freqs) > 0:
        nmax = int(np.max(freqs) // f0)
        for n in range(1, max(nmax, 1) + 1):
            ax.axvline(n * f0, color="gray", linestyle="--", alpha=0.35, linewidth=0.8)

    ax.set(
        title=f"Periodograma do resíduo - TIC {tic_id}" if tic_id else "Periodograma do resíduo",
        xlabel="Frequência (1/dia)", ylabel="Potência",
    )
    sfx = _norm_suffix(filename_suffix)
    caminho = os.path.join(_tic_dir(int(tic_id) if tic_id is not None else 0, cfg, subdir), f"periodograma_residuo{sfx}.png")
    _finalizar_fig(fig, cfg, save_path=caminho)
    return fig

# %% [markdown]
# Pipeline de análise para um alvo

# %%
def analisar_exoplaneta_alvo(
    tic_id: int,
    period_days: float,
    setores_disponiveis: Iterable[int] | None,
    cfg: Config,
) -> Dict[str, plt.Figure]:
    """
    Executa o pipeline para um alvo, com controle de composição por setor.
    """
    figs: Dict[str, plt.Figure] = {}

    setores = cfg.sectors if cfg.sectors is not None else setores_disponiveis
    curva_unica, curvas_por_setor = baixar_curvas_por_setor(int(tic_id), setores)

    # Se nada foi baixado, retorna vazio
    if curva_unica is None and not curvas_por_setor:
        return figs

    f0 = 1.0 / float(period_days)

    def _rodar_em_curva(curva: lk.LightCurve, sfx_key: str = "", sfx_file: str = "", subdir: Optional[str] = None) -> None:
        # Periodograma + harmônicos
        harmonicos, fmax, n_harm, pg, freqs, powers = detectar_harmonicos_com_expansao(
            curva=curva, f0=f0, max_harm_ini=5, step=10, limite=cfg.max_harm_limit,
            pctl=cfg.pctl, tol=cfg.tol, oversample=cfg.oversample
        )
        # Ajuste Fourier (≥ 1 harmônico)
        num_h = max(len(harmonicos), 1)
        params = ajustar_fourier(curva, f0=f0, num_harm=num_h)

        if _should("transits_full", cfg):
            figs[f"transits_full{sfx_key}"] = plot_transits_full(curva, tic_id, cfg, filename_suffix=sfx_file, subdir=subdir)

        if _should("periodogram", cfg):
            period_dom = getattr(pg, "period_at_max_power", None)
            period_dom = float(period_dom.value) if period_dom is not None else None
            figs[f"periodogram{sfx_key}"] = plot_periodogram(
                freqs, powers, f0, period_dom, harmonicos, int(tic_id), cfg, max_harm=n_harm, filename_suffix=sfx_file, subdir=subdir
            )

        if _should("full_with_model", cfg):
            figs[f"full_with_model{sfx_key}"] = plot_full_with_model(curva, params, f0, int(tic_id), cfg, filename_suffix=sfx_file, subdir=subdir)

        if _should("transit_folded", cfg) or _should("folded_with_model", cfg):
            fig_fold, t0_est = plot_transit_folded(curva, params, f0, int(tic_id), cfg, t0=cfg.t0_fold, filename_suffix=sfx_file, subdir=subdir)
            if _should("transit_folded", cfg):
                figs[f"transit_folded{sfx_key}"] = fig_fold
            if _should("folded_with_model", cfg):
                figs[f"folded_with_model{sfx_key}"] = fig_fold

        # Opcional: análise do resíduo
        # t, resid, _ = calcular_residuos(curva, params, f0)
        # freqs_r, powers_r, _ = periodograma_residuo(t, resid, max_frequency=fmax, oversample=cfg.oversample)
        # plot_periodograma_residuo(freqs_r, powers_r, f0=f0, tic_id=int(tic_id), cfg=cfg, filename_suffix=sfx_file, subdir=subdir)

    # ===== Modo combined =====
    if cfg.sector_mode in ("combined", "both"):
        if cfg.sector_overlay and curvas_por_setor:
            # Sobreposição de curvas por setor no mesmo eixo (figura visual)
            figs["transits_full_overlay"] = plot_transits_full_overlay(curvas_por_setor, int(tic_id), cfg, subdir="overlay")
            # Para a análise numérica, usa curva unificada se existir; senão, faz append manual
            curva_para_analise = None
            if curva_unica is not None:
                curva_para_analise = curva_unica
            else:
                try:
                    curva_para_analise = curvas_por_setor[0][1].append([lc for _, lc in curvas_por_setor[1:]])
                except Exception:
                    curva_para_analise = curvas_por_setor[0][1]
            _rodar_em_curva(curva_para_analise, sfx_key="_combined", sfx_file="combined", subdir="combined")
        else:
            # Sem overlay: apenas a curva unificada
            if curva_unica is not None:
                _rodar_em_curva(curva_unica, sfx_key="_combined", sfx_file="combined", subdir="combined")

    # ===== Modo separate =====
    if cfg.sector_mode in ("separate", "both") and curvas_por_setor:
        for setor, lc in curvas_por_setor:
            sfx_key = f"_S{setor}"
            sfx_file = f"S{setor}"
            _rodar_em_curva(lc, sfx_key=sfx_key, sfx_file=sfx_file, subdir=f"S{setor}")

    return figs

# %% [markdown]
# Execução a partir de CSV (ExoFOP)

# %%
def rodar_de_csv(cfg: Config) -> None:
    """
    Aplica o pipeline a todos os alvos listados em um CSV do ExoFOP.
    """
    # Força modo interativo conforme cfg.show_plots e evita auto-show indesejado
    try:
        if cfg.show_plots:
            plt.ion()
            if cfg.verbose:
                print("[INFO] Modo interativo ON (plt.ion())")
        else:
            plt.ioff()
            if cfg.verbose:
                print("[INFO] Modo interativo OFF (plt.ioff())")
        # Evita warnings de muitas figuras abertas
        import matplotlib as mpl
        mpl.rcParams['figure.max_open_warning'] = 0
    except Exception as e:
        print(f"[WARN] Não foi possível ajustar modo interativo: {e}")

    assert cfg.csv_path is not None, "csv_path não definido na Config"

    try:
        tabela = pd.read_csv(cfg.csv_path)
    except Exception as e:
        raise SystemExit(f"Falha ao ler CSV: {e}")

    for _, linha in tabela.iterrows():
        try:
            tic_id = int(linha.get("star_name"))
        except Exception:
            continue

        if cfg.tic_only is not None and tic_id != int(cfg.tic_only):
            continue

        try:
            period_days = float(linha.get("orbital_period(days)"))
        except Exception:
            continue

        setores_disp: List[int] = []
        for s in str(linha.get("Sectors", "")).split(","):
            try:
                setores_disp.append(int(s.strip()))
            except ValueError:
                pass
        setores_disp = sorted(set(setores_disp)) if setores_disp else None

        if cfg.verbose:
            print(f"[INFO] TIC {tic_id} | period={period_days} d | setores={setores_disp} | modo={cfg.sector_mode}")

        analisar_exoplaneta_alvo(tic_id, period_days, setores_disp, cfg)

# %% [markdown]
# Parâmetros rápidos e execução (ajuste conforme necessário no Spyder)

# %%
if __name__ == "__main__":
    cfg = Config(
        csv_path="/home/joshua/Documentos/ufrgs/light-curves-main/dados_exoplanetas/alvos_kp_taina.csv",
        #csv_path="/home/joshua/Documentos/ufrgs/light-curves-main/dados_exoplanetas/data_ExoFOP(taina).csv",
        tic_only=25155310,            # None para rodar todos do CSV
        sectors=[1],            # None para usar todos os setores disponíveis do alvo
        sector_mode="separate",           # "combined" | "separate" | "both"
        sector_overlay=True,          # no modo combined, fazer sobreposição por setor
        which_plots=(
            "transits_full",
            "transit_folded",
            "periodogram",
            "full_with_model",
            "folded_with_model",
        ),
        save=False,
        outdir="/home/joshua/Documentos/ufrgs/light-curves-main/exemplos/exemplos_taina/",  # caminho absoluto recomendado
        fig_dpi=100,
        pctl=87.0,
        tol=0.03,
        max_harm_limit=150,
        oversample=15,
        t0_fold=None,
        show_plots=True,
        verbose=True,
    )

    rodar_de_csv(cfg)
