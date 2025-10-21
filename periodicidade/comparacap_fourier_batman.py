# %% ----------------------- IMPORTS -----------------------
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

# Modelo físico de trânsito
try:
    import batman
except ImportError:
    raise SystemExit("A biblioteca 'batman' não está instalada. Instale com: pip install batman-package")

# %% ----------------------- 1) CURVA ARTIFICIAL (BOX + SUBTRÂNSITO) + RUÍDO -----------------------
def gerar_curva_box_com_sub(
    dias_total=27.0,
    cad_min=2.0,
    # Trânsito principal:
    P1=3.14159, t01=0.5, depth1_ppm=1500.0, dur1_horas=3.0,
    # Subtrânsito (segundo planeta):
    P2=2.20, t02=0.3, depth2_ppm=600.0, dur2_horas=2.0,
    # Ruído:
    ruido_ppm=300.0,
    seed=42
):
    """
    Curva sintética realista:
      - Trânsitos físicos (BATMAN) para 1–2 planetas (limb darkening quadrático).
      - Variabilidade estelar = combinação de senoides (rotação + harmônicos) em ppm.
      - Ruído gaussiano branco (ppm).
    Baseline final ~ 0 ppm (retorno em ppm).
    """
    rng = np.random.default_rng(seed)
    dt = cad_min / (60*24)
    t  = np.arange(0.0, dias_total, dt)

    # ---------- helper: parâmetros BATMAN a partir de depth e duração ----------
    def _params_from_depth_duration(P, t0, depth_ppm, dur_horas, b=0.10, u=(0.3, 0.2)):
        depth_frac = max(depth_ppm, 1e-12) * 1e-6
        rp = float(np.sqrt(depth_frac))
        T = float(dur_horas) / 24.0  # dias
        root = max((1.0 + rp)**2 - b**2, 1e-10)
        aRs = (P / (np.pi * T)) * np.sqrt(root)
        cosi = float(np.clip(b / max(aRs, 1e-10), -1.0, 1.0))
        inc_deg = float(np.degrees(np.arccos(cosi)))

        params = batman.TransitParams()
        params.t0 = float(t0)
        params.per = float(P)
        params.rp  = rp
        params.a   = float(aRs)
        params.inc = inc_deg
        params.ecc = 0.0
        params.w   = 90.0
        params.u   = [0.3, 0.2]
        params.limb_dark = "quadratic"
        return params

    # ---------- trânsitos (fluxo relativo ~1 fora do trânsito) ----------
    params1 = _params_from_depth_duration(P1, t01, depth1_ppm, dur1_horas)
    model1  = batman.TransitModel(params1, t)
    flux1   = model1.light_curve(params1)

    params2 = _params_from_depth_duration(P2, t02, depth2_ppm, dur2_horas)
    model2  = batman.TransitModel(params2, t)
    flux2   = model2.light_curve(params2)

    # Combinação física correta de múltiplos trânsitos: multiplicativa
    flux_transit_rel = flux1 * flux2  # relativo

    # ---------- variabilidade estelar senoidal (ppm) ----------
    # Prot ~ 10 d + harmônicos + termo mais lento; fases aleatórias mas reprodutíveis (seed)
    Prot = 10.0
    Pslow = 40.0
    amps_ppm = np.array([400.0, 180.0, 90.0, 60.0])  # amplitudes dos termos
    periods  = np.array([Prot, Prot/2, Prot/3, Pslow])

    phases = rng.uniform(0, 2*np.pi, size=amps_ppm.size)
    var_ppm = np.zeros_like(t, dtype=float)
    for A, P, phi in zip(amps_ppm, periods, phases):
        var_ppm += A * np.cos(2*np.pi*t/P + phi)

    # Converte variabilidade para fluxo relativo multiplicativo: 1 + var/1e6
    flux_var_rel = 1.0 + (var_ppm * 1e-6)

    # ---------- fluxo total relativo e em ppm ----------
    flux_total_rel = flux_transit_rel * flux_var_rel
    flux_ppm = (flux_total_rel - 1.0) * 1e6  # baseline ~ 0 ppm

    # ---------- ruído branco (ppm) ----------
    flux_ppm += rng.normal(0.0, ruido_ppm, size=flux_ppm.size)

    meta = {
        "P1": P1, "f01": 1.0/P1, "depth1_ppm": depth1_ppm, "dur1_h": dur1_horas, "t01": t01,
        "P2": P2, "f02": 1.0/P2, "depth2_ppm": depth2_ppm, "dur2_h": dur2_horas, "t02": t02,
        "ruido_ppm": ruido_ppm,
        "variabilidade": {
            "amps_ppm": amps_ppm.tolist(),
            "periods_d": periods.tolist()
        }
    }
    return lk.LightCurve(time=t, flux=flux_ppm), meta

# %% ----------------------- 2) FOURIER (design + ajuste) -----------------------
def design_matrix(t, f0, N_HARM):
    cols = [np.ones_like(t)]
    for n in range(1, N_HARM+1):
        cols.append(np.cos(2*np.pi*n*f0*t))
        cols.append(np.sin(2*np.pi*n*f0*t))
    return np.column_stack(cols)

def ajustar_fourier_linear(lc, f0, N_HARM=40):
    t = lc.time.value
    y = lc.flux.value
    X = design_matrix(t, f0, N_HARM)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_fit = X @ coef
    return coef, y_fit

# %% ----------------------- 3) UTILITÁRIOS: folded e BATMAN -----------------------
def folded_with_model(t, flux, model, P, t0=0.0):
    """Retorna (fase ordenada, dados ordenados, modelo ordenado) para o período P."""
    phase = ((t - t0 + 0.5*P) % P) / P - 0.5
    idx = np.argsort(phase)
    return phase[idx], flux[idx], model[idx]

def fold_series(t, series, P, t0=0.0):
    """Fold simples para uma única série (ex.: diferença de modelos)."""
    phase = ((t - t0 + 0.5*P) % P) / P - 0.5
    idx = np.argsort(phase)
    return phase[idx], series[idx]

def estimar_parametros_batman(meta, b=0.1, u=(0.3, 0.2)):
    """
    Estima parâmetros físicos para o BATMAN a partir de profundidade e duração.
    Aproximações:
      - rp ~ sqrt(depth_frac), depth_frac = depth_ppm * 1e-6
      - a/Rs via fórmula aproximada da duração com impacto b fixo
      - inclinação via b = (a/Rs) cos i
    """
    P = float(meta["P1"])
    T = float(meta["dur1_h"])/24.0          # duração (dias)
    depth_ppm = float(meta["depth1_ppm"])
    t0 = float(meta["t01"])

    depth_frac = max(depth_ppm, 1e-12) * 1e-6
    rp = np.sqrt(depth_frac)

    root = max((1.0 + rp)**2 - b**2, 1e-8)
    aRs = (P / (np.pi * T)) * np.sqrt(root)

    cosi = np.clip(b / max(aRs, 1e-8), -1.0, 1.0)
    inc = np.degrees(np.arccos(cosi))

    u1, u2 = u
    params = batman.TransitParams()
    params.t0 = t0
    params.per = P
    params.rp  = rp
    params.a   = aRs
    params.inc = float(inc)
    params.ecc = 0.0
    params.w   = 90.0
    params.u   = [u1, u2]
    params.limb_dark = "quadratic"
    return params

# %% ----------------------- 4) PIPELINE PRINCIPAL -----------------------
if __name__ == "__main__":
    # (A) Dados sintéticos
    lc, meta = gerar_curva_box_com_sub(
        P1=3.14159,  t01=0.5,  depth1_ppm=1500, dur1_horas=3.0,
        P2=5.20,     t02=0.3,  depth2_ppm=800,  dur2_horas=2.0,
        ruido_ppm=250
    )
    t = lc.time.value
    y = lc.flux.value  # ppm, baseline ~ 0

    # (B) Modelo Fourier (trânsito principal)
    f0 = meta["f01"]
    N_HARM = 40
    _, y_fit_fourier = ajustar_fourier_linear(lc, f0=f0, N_HARM=N_HARM)

    # (C) Modelo BATMAN (trânsito principal, convertido para ppm)
    params_bm = estimar_parametros_batman(meta, b=0.1, u=(0.3, 0.2))
    bm = batman.TransitModel(params_bm, t)       # fluxo relativo ~ 1
    flux_rel = bm.light_curve(params_bm)         # relativo
    y_batman_ppm = (flux_rel - 1.0) * 1e6        # ppm, baseline 0

    # 0) Série temporal — curva completa + modelos
    plt.figure(figsize=(12,4), dpi=200)
    plt.plot(t, y, '.', ms=1.8, alpha=0.45, label='Dados (ppm)')
    #plt.plot(t, y_fit_fourier, '-', lw=1.6, label='Modelo Fourier (ppm)')
    #plt.plot(t, y_batman_ppm, '-', lw=1.6, label='Modelo BATMAN (ppm)')
    plt.xlabel('Tempo (dias)'); plt.ylabel('Fluxo (ppm)')
    plt.title('Curva de luz completa (domínio do tempo)')
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    # (Opcional) Resíduos no tempo
    res_F = y - y_fit_fourier
    res_B = y - y_batman_ppm
    plt.figure(figsize=(12,4), dpi=200)
    plt.plot(t, res_F, '-', lw=1.0, alpha=0.9, label='Resíduos vs Fourier')
    plt.plot(t, res_B, '-', lw=1.0, alpha=0.9, label='Resíduos vs BATMAN')
    plt.axhline(0, lw=1)
    plt.xlabel('Tempo (dias)'); plt.ylabel('Fluxo (ppm)')
    plt.title('Resíduos no domínio do tempo')
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    # ==================== GRÁFICOS SEPARADOS ====================

    # 1) Folded — Modelo Fourier
    phase_F, y_F, yfit_F = folded_with_model(t, y, y_fit_fourier, meta["P1"], t0=meta["t01"])
    plt.figure(figsize=(10,4), dpi=200)
    plt.plot(phase_F, y_F, '.', ms=2, alpha=0.35, label='Dados')
    plt.plot(phase_F, yfit_F, '-', lw=2, label='Modelo Fourier', c='red')
    plt.xlabel('Fase (P1)'); plt.ylabel('Fluxo (ppm)')
    plt.title('Trânsito principal (Modelo Fourier)')
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    # 2) Folded — Modelo BATMAN
    phase_B, y_B, ybat_B = folded_with_model(t, y, y_batman_ppm, meta["P1"], t0=meta["t01"])
    plt.figure(figsize=(10,4), dpi=200)
    plt.plot(phase_B, y_B, '.', ms=2, alpha=0.35, label='Dados')
    plt.plot(phase_B, ybat_B, '-', lw=2, label='Modelo BATMAN', c='red')
    plt.xlabel('Fase (P1)'); plt.ylabel('Fluxo (ppm)')
    plt.title('Trânsito principal (Modelo BATMAN)')
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    # 3) Folded — Diferença de modelos (Fourier − BATMAN)
    diff_model = y_fit_fourier - y_batman_ppm
    phase_D, diff_D = fold_series(t, diff_model, meta["P1"], t0=meta["t01"])
    plt.figure(figsize=(10,4), dpi=200)
    plt.plot(phase_D, diff_D, '-', lw=2, label='Fourier − BATMAN')
    plt.axhline(0, lw=1)
    plt.xlabel('Fase (P1)'); plt.ylabel('Fluxo (ppm)')
    plt.title('Diferença entre modelos (Fourier − BATMAN)')
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
