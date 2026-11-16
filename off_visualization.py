"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import off_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend_y=-0.22):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _zeigen(p, n):
    """Die ersten n Werte eines Verlaufs (Anzeige; die Kennzahlen kommen aus dem ganzen Lauf)."""
    return list(range(min(n, len(p)))), [float(v) for v in p[:n]]


def build_ort(res, s, n_zeigen: int) -> go.Figure:
    """Ort p_k mit und ohne Störungsmodell, das Ziel r und der Dauerfehler-Wert ohne Störungsmodell; mit Stellgrenze die abgeschnittenen Läufe."""
    fig = go.Figure()
    if s.umax is not None and res.get("grenze") is not None:
        po, pm = res["g_ohne"]["p"], res["g_mit"]["p"]
        n = min(n_zeigen, len(po))
        x = list(range(n))
        fig.add_trace(go.Scatter(x=x, y=[float(v) for v in po[:n]], mode="lines", name="ohne Störungsmodell (mit Stellgrenze)", line=dict(color=C.COLOR_OHNE, width=2)))
        fig.add_trace(go.Scatter(x=x, y=[float(v) for v in pm[:n]], mode="lines", name="mit Störungsmodell (mit Stellgrenze)", line=dict(color=C.COLOR_MIT, width=3)))
    else:
        x, yo = _zeigen(res["ohne"]["p"], n_zeigen)
        _, ym = _zeigen(res["mit"]["p"], n_zeigen)
        fig.add_trace(go.Scatter(x=x, y=yo, mode="lines", name="ohne Störungsmodell (u = −K x)", line=dict(color=C.COLOR_OHNE, width=2),
                                 hovertemplate="Schritt %{x}: %{y:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=ym, mode="lines", name="mit Störungsmodell und Beobachter", line=dict(color=C.COLOR_MIT, width=3),
                                 hovertemplate="Schritt %{x}: %{y:.3f}<extra></extra>"))
        if abs(s.b) > 1e-12:
            fig.add_trace(go.Scatter(x=x, y=[v + s.b for v in ym], mode="lines", name="gemessen (Ort + Messbias), mit Störungsmodell",
                                     line=dict(color=C.COLOR_SCHAETZ, width=2, dash="dot")))
        fest = res["ohne"]["p_fest"]
        if np.isfinite(fest) and abs(fest - s.r) > 1e-9:
            fig.add_hline(y=fest, line=dict(color=C.COLOR_OHNE, width=1, dash="dash"), annotation_text=f"ohne Modell bleibt: {fest:.3f}".replace(".", ","), annotation_position="top right")
    fig.add_hline(y=s.r, line=dict(color=C.COLOR_ZIEL, width=1, dash="dot"), annotation_text="Ziel r", annotation_position="bottom right")
    fig.update_xaxes(title_text="Periode k")
    fig.update_yaxes(title_text="Ort p")
    return _base(fig)


def build_schaetzung(res, s, n_zeigen: int) -> go.Figure:
    """Geschätzte Störung d̂_k gegen die wahre Störung d (mit Stellgrenze: aus dem abgeschnittenen Lauf)."""
    if s.umax is not None and res.get("grenze") is not None:
        dh = res["g_mit"]["dh"]
    else:
        dh = res["mit"]["dh"]
    x, y = _zeigen(dh, n_zeigen)
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", name="geschätzte Störung d̂", line=dict(color=C.COLOR_SCHAETZ, width=3), hovertemplate="Schritt %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=s.d, line=dict(color=C.COLOR_ZIEL, width=1, dash="dot"), annotation_text="wahre Störung d", annotation_position="bottom right")
    fig.update_xaxes(title_text="Periode k")
    fig.update_yaxes(title_text="Störung")
    return _base(fig)


def build_dauerfehler_R(rows) -> go.Figure:
    """p_ss/d über R (Wagen) und der Wert 1,618 (goldener Schnitt, nur Zahlenvergleich)."""
    from off_evaluation import PHI

    xs = [r["R"] for r in rows]
    fig = go.Figure(go.Scatter(x=xs, y=[r["p_ss"] for r in rows], mode="lines+markers", name="p_ss / d = 1/K₁", line=dict(color=C.COLOR_OHNE, width=3),
                               hovertemplate="R = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=PHI, line=dict(color=C.COLOR_ZIEL, width=1, dash="dash"), annotation_text="1,618", annotation_position="bottom right")
    fig.update_xaxes(title_text="Stellgewicht R", type="log")
    fig.update_yaxes(title_text="Dauerfehler je Einheit d")
    return _base(fig, height=300)


def build_beobachter(zeilen, ohne, rho_sel) -> go.Figure:
    """Spitze, Abklingzeit und Rauschen der Stellgröße über dem Beobachterpol (logarithmische Achse)."""
    xs = [z["rho"] for z in zeilen]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[z["spitze"] for z in zeilen], mode="lines+markers", name="Spitze von |p|", line=dict(color=C.COLOR_OHNE, width=3)))
    fig.add_trace(go.Scatter(x=xs, y=[z["std_u"] for z in zeilen], mode="lines+markers", name="Rauschen der Stellgröße std(u) je std(n)", line=dict(color=C.COLOR_SCHAETZ, width=3)))
    fig.add_trace(go.Scatter(x=xs, y=[z["abkling"] for z in zeilen], mode="lines+markers", name="Abklingzeit (Perioden)", line=dict(color=C.COLOR_ZWEI, width=2, dash="dot")))
    fig.add_hline(y=ohne["spitze"], line=dict(color=C.COLOR_ZIEL, width=1, dash="dash"), annotation_text="Spitze ohne Beobachter", annotation_position="top left")
    fig.add_vline(x=rho_sel, line=dict(color=C.COLOR_MIT, width=1, dash="dot"))
    fig.update_xaxes(title_text="Beobachterpol ρ (dreifach)", range=[0, 1])
    fig.update_yaxes(title_text="Wert (logarithmisch)", type="log")
    return _base(fig, height=380)


def build_rausch(laeufe) -> go.Figure:
    """Stellgröße bei Messrauschen (gleiche Rauschfolge für alle Pole)."""
    fig = go.Figure()
    farben = [C.COLOR_OHNE, C.COLOR_MIT, C.COLOR_SCHAETZ, C.COLOR_ZWEI]
    for i, (rho, v) in enumerate(laeufe.items()):
        x, y = _zeigen(v["u"], len(v["u"]))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"ρ = {rho:g}".replace(".", ","), line=dict(color=farben[i % 4], width=1.5)))
    fig.update_xaxes(title_text="Periode k")
    fig.update_yaxes(title_text="Stellgröße u")
    return _base(fig, height=300)


def build_gebiet(zeilen, gebiet_ohne, g_sel, rho_sel) -> go.Figure:
    """Gebiet der stabilen Streckenverstärkung g über dem Beobachterpol: je ρ ein Balken von g_min bis g_max, dazu das Gebiet ohne Beobachter und der gewählte Wert g."""
    fig = go.Figure()
    for z in zeilen:
        lo, hi = z["gebiet"]
        fig.add_trace(go.Bar(y=[f"ρ = {z['rho']:g}".replace(".", ",")], x=[hi - lo], base=[lo], orientation="h", showlegend=False, marker_color=C.COLOR_MIT if abs(z["rho"] - rho_sel) < 1e-9 else "rgba(47,107,101,0.45)",
                             hovertemplate=f"{lo:.3f} bis {hi:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(y=["ohne Beobachter"], x=[gebiet_ohne[1] - gebiet_ohne[0]], base=[gebiet_ohne[0]], orientation="h", showlegend=False, marker_color="rgba(192,80,77,0.6)",
                         hovertemplate=f"{gebiet_ohne[0]:.3f} bis {gebiet_ohne[1]:.3f}<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color=C.COLOR_ZIEL, width=1, dash="dot"), annotation_text="Entwurf g = 1", annotation_position="top")
    fig.add_vline(x=g_sel, line=dict(color=C.COLOR_ZWEI, width=2), annotation_text="gewähltes g", annotation_position="bottom")
    fig.update_xaxes(title_text="wahre Streckenverstärkung g (stabil im Balken)", range=[0, 4.5])
    fig.update_layout(barmode="overlay")
    return _base(fig, height=380)


def build_zwei_wege(integral, beob, T) -> go.Figure:
    """Ort nach dem Störsprung d = 1: Integralzustand gegen Störungsbeobachter (Zustand gemessen)."""
    n = min(T, len(integral["p"]), len(beob["p"]))
    x = list(range(n))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[float(v) for v in integral["p"][:n]], mode="lines", name="Integralzustand", line=dict(color=C.COLOR_ZWEI, width=3)))
    fig.add_trace(go.Scatter(x=x, y=[float(v) for v in beob["p"][:n]], mode="lines", name="Störungsbeobachter", line=dict(color=C.COLOR_MIT, width=3)))
    fig.add_hline(y=0.0, line=dict(color=C.COLOR_ZIEL, width=1, dash="dot"))
    fig.update_xaxes(title_text="Periode k")
    fig.update_yaxes(title_text="Ort p")
    return _base(fig)


def build_rampe(pm, po, n: int | None = None) -> go.Figure:
    """Ort bei der Rampe d_k = s k: mit Beobachter wird er konstant, ohne läuft er weg."""
    n = len(pm) if n is None else min(n, len(pm))
    x = list(range(n))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[float(v) for v in po[:n]], mode="lines", name="ohne Störungsmodell", line=dict(color=C.COLOR_OHNE, width=2)))
    fig.add_trace(go.Scatter(x=x, y=[float(v) for v in pm[:n]], mode="lines", name="mit Störungsmodell", line=dict(color=C.COLOR_MIT, width=3)))
    fig.update_xaxes(title_text="Periode k")
    fig.update_yaxes(title_text="Ort p")
    return _base(fig, height=300)


def build_frequenzgang(ws, kurve, kurve0, w_sel) -> go.Figure:
    """|p/d| über der Kreisfrequenz (Sinusstörung) mit und ohne Beobachter."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(ws), y=[float(v) for v in kurve0], mode="lines", name="ohne Störungsmodell", line=dict(color=C.COLOR_OHNE, width=2)))
    fig.add_trace(go.Scatter(x=list(ws), y=[float(v) for v in kurve], mode="lines", name="mit Störungsmodell", line=dict(color=C.COLOR_MIT, width=3)))
    fig.add_vline(x=w_sel, line=dict(color=C.COLOR_ZWEI, width=1, dash="dot"))
    fig.update_xaxes(title_text="Kreisfrequenz ω (Bogenmaß je Periode)", type="log")
    fig.update_yaxes(title_text="|p / d|", type="log")
    return _base(fig, height=300)


def build_stellgrenze(zeilen) -> go.Figure:
    """Ort mit Stellgrenze |u| ≤ 1 für d = 0,3 / 0,9 / 1,5, mit Störungsmodell (durchgezogen) und ohne (gestrichelt)."""
    fig = go.Figure()
    farben = [C.COLOR_MIT, C.COLOR_SCHAETZ, C.COLOR_OHNE]
    for i, z in enumerate(zeilen):
        n = 120
        x = list(range(n))
        fig.add_trace(go.Scatter(x=x, y=[float(v) for v in z["mit"]["p"][:n]], mode="lines", name=f"d = {z['d']:g}, mit Modell".replace(".", ","), line=dict(color=farben[i], width=3)))
        fig.add_trace(go.Scatter(x=x, y=[float(v) for v in z["ohne"]["p"][:n]], mode="lines", name=f"d = {z['d']:g}, ohne Modell".replace(".", ","), line=dict(color=farben[i], width=2, dash="dash")))
    fig.update_xaxes(title_text="Periode k")
    fig.update_yaxes(title_text="Ort p (Start 4)", range=[-5, 60])
    return _base(fig, height=380, legend_y=-0.3)
