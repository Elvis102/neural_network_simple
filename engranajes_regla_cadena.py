"""
engranajes_regla_cadena.py
Visualización interactiva: regla de la cadena como sistema de engranajes.

z = (f ∘ g)(x) = f(g(x))
dz/dx = (dz/dy) · (dy/dx)

Uso en Quarto (.qmd con jupyter: python3):
    ```{python}
    from engranajes_regla_cadena import crear_visualizacion
    crear_visualizacion().show()
    ```

Uso directo:
    python engranajes_regla_cadena.py
"""

import numpy as np
import plotly.graph_objects as go


# ═══════════════════════════════════════════════════════════════════
# Geometría de engranajes
# ═══════════════════════════════════════════════════════════════════

def gear_outline(cx, cy, n_teeth, r_pitch, tooth_h, rotation=0, npts=500):
    """Contorno de un engranaje con perfil trapezoidal suavizado.

    cx, cy     : centro del engranaje
    n_teeth    : número de dientes
    r_pitch    : radio primitivo (donde los dientes engranan)
    tooth_h    : altura del diente (mitad por encima, mitad por debajo del pitch)
    rotation   : ángulo de rotación en radianes
    """
    theta = np.linspace(0, 2 * np.pi, npts, endpoint=True)
    # Sinusoide recortada → dientes trapezoidales
    wave = np.clip(2.5 * np.sin(n_teeth * (theta - rotation)), -1, 1)
    r = r_pitch + tooth_h * wave
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


def hub_circle(cx, cy, r_hub, npts=50):
    """Pequeño círculo central (eje del engranaje)."""
    theta = np.linspace(0, 2 * np.pi, npts, endpoint=True)
    return cx + r_hub * np.cos(theta), cy + r_hub * np.sin(theta)


def marker_line(cx, cy, r_pitch, angle):
    """Línea radial que indica la posición angular actual."""
    r_end = r_pitch * 0.82
    return ([cx, cx + r_end * np.cos(angle)],
            [cy, cy + r_end * np.sin(angle)])


# ═══════════════════════════════════════════════════════════════════
# Visualización principal
# ═══════════════════════════════════════════════════════════════════

def crear_visualizacion():
    """
    Tres engranajes representando z = f(g(x)).

    Engranaje x  : 10 dientes, r = 1.5  (entrada)
    Engranaje g  : 15 dientes, r = 2.25 (intermedio, y = g(x))
    Engranaje f  : 20 dientes, r = 3.0  (salida, z = f(y))

    Razones de transmisión (análogas a derivadas):
        dy/dx = n₁/n₂ = 10/15 = 2/3
        dz/dy = n₂/n₃ = 15/20 = 3/4
        dz/dx = dy/dx × dz/dy = 2/3 × 3/4 = 1/2

    ¿Cuánto gira x para que z dé 1/4 vuelta?
        θ_x = (1/4) / (1/2) = 1/2 vuelta  ✓
    """

    # ── Parámetros de los engranajes ──
    n1, n2, n3 = 10, 15, 20           # dientes
    pitch_k = 0.15                     # r = pitch_k × n
    r1, r2, r3 = pitch_k * n1, pitch_k * n2, pitch_k * n3  # 1.5, 2.25, 3.0
    tooth_h = 0.12                     # altura del diente

    # Centros (alineados horizontalmente, tangentes en los radios primitivos)
    cx1, cy1 = 0.0, 0.0
    cx2, cy2 = r1 + r2, 0.0           # 3.75
    cx3, cy3 = cx2 + r2 + r3, 0.0     # 9.0

    # Razones de transmisión
    ratio_12 = n1 / n2    # dy/dx = 2/3
    ratio_23 = n2 / n3    # dz/dy = 3/4
    ratio_13 = n1 / n3    # dz/dx = 1/2

    # Colores: (línea, relleno, hub)
    COLORS = {
        'x': ('darkorange',  'rgba(255,165,0,0.20)',  'rgba(255,165,0,0.55)'),
        'g': ('royalblue',   'rgba(65,105,225,0.20)', 'rgba(65,105,225,0.55)'),
        'f': ('seagreen',    'rgba(46,139,87,0.20)',  'rgba(46,139,87,0.55)'),
    }
    keys = ['x', 'g', 'f']
    cxs  = [cx1, cx2, cx3]
    cys  = [cy1, cy2, cy3]
    rs   = [r1, r2, r3]
    ns   = [n1, n2, n3]

    # ── Funciones auxiliares ──

    def calc_rotations(theta):
        """Ángulos de rotación de los tres engranajes para un θ de entrada.

        Condición de engrane: en el punto de contacto, cuando un engranaje
        tiene un diente (cresta), el otro debe tener un valle (hueco).
        El offset π/n desplaza la fase medio diente para lograr esto.
        """
        rot1 = theta
        rot2 = -theta * ratio_12 + np.pi / n2   # opuesto + offset engrane 1↔2
        rot3 = theta * ratio_13 + np.pi / n3    # mismo sentido + offset engrane 2↔3
        return rot1, rot2, rot3

    def build_data(theta):
        """Genera las coordenadas de los 9 elementos gráficos."""
        rots = calc_rotations(theta)
        items = []
        # 0-2: contornos de engranajes
        for i in range(3):
            gx, gy = gear_outline(cxs[i], cys[i], ns[i], rs[i], tooth_h, rots[i])
            items.append((gx, gy))
        # 3-5: líneas marcadoras
        for i in range(3):
            mx, my = marker_line(cxs[i], cys[i], rs[i], rots[i])
            items.append((mx, my))
        # 6-8: centros (hubs)
        for i in range(3):
            hx, hy = hub_circle(cxs[i], cys[i], rs[i] * 0.14)
            items.append((hx, hy))
        return items

    def build_annotations(theta):
        """Anotaciones dinámicas: vueltas de cada engranaje.

        Nota: NO usar $...$ (MathJax) dentro de Plotly en Quarto,
        porque Quarto intercepta el MathJax y rompe el widget.
        Usar HTML + Unicode en su lugar.
        """
        vx = theta / (2 * np.pi)
        vy = theta * ratio_12 / (2 * np.pi)
        vz = theta * ratio_13 / (2 * np.pi)
        return [
            # Etiquetas debajo de cada engranaje
            dict(x=cx1, y=-r1 - 0.65, xref='x', yref='y',
                 text=f"<b><i>x</i></b><br>{vx:.2f} vueltas",
                 showarrow=False, font=dict(size=14, color='darkorange'),
                 align='center'),
            dict(x=cx2, y=-r2 - 0.65, xref='x', yref='y',
                 text=f"<b><i>y</i> = <i>g</i>(<i>x</i>)</b><br>{vy:.2f} vueltas",
                 showarrow=False, font=dict(size=14, color='royalblue'),
                 align='center'),
            dict(x=cx3, y=-r3 - 0.65, xref='x', yref='y',
                 text=f"<b><i>z</i> = <i>f</i>(<i>y</i>)</b><br>{vz:.2f} vueltas",
                 showarrow=False, font=dict(size=14, color='seagreen'),
                 align='center'),
            # Número de dientes arriba
            dict(x=cx1, y=r1 + 0.55, xref='x', yref='y',
                 text=f"<i>{n1} dientes</i>",
                 showarrow=False, font=dict(size=11, color='gray')),
            dict(x=cx2, y=r2 + 0.55, xref='x', yref='y',
                 text=f"<i>{n2} dientes</i>",
                 showarrow=False, font=dict(size=11, color='gray')),
            dict(x=cx3, y=r3 + 0.55, xref='x', yref='y',
                 text=f"<i>{n3} dientes</i>",
                 showarrow=False, font=dict(size=11, color='gray')),
            # Caja: Regla de la cadena
            dict(x=(cx1 + cx3) / 2, y=-r3 - 2.0, xref='x', yref='y',
                 text=("<b>Regla de la cadena:</b><br>"
                       "<i>dz/dx</i> = <i>dz/dy</i> · <i>dy/dx</i>"
                       f" = ({n2}/{n3}) · ({n1}/{n2})"
                       f" = <b>{n1}/{n3} = {ratio_13:.1f}</b>"),
                 showarrow=False, font=dict(size=13),
                 bgcolor='lightyellow', bordercolor='gray',
                 borderwidth=1, borderpad=8, align='center'),
            # Pregunta pedagógica
            dict(x=(cx1 + cx3) / 2, y=-r3 - 3.5, xref='x', yref='y',
                 text=("¿Cuánto debe girar <b><i>x</i></b> para que "
                       "<b><i>z</i></b> dé <b>¼ de vuelta</b>?<br>"
                       "Respuesta: <i>θ<sub>x</sub></i> = (¼) ÷ (½) = "
                       "<b>½ vuelta</b>"),
                 showarrow=False, font=dict(size=12),
                 bgcolor='rgba(255,228,196,0.5)', bordercolor='darkorange',
                 borderwidth=1, borderpad=6, align='center'),
        ]

    # ── Crear la figura ──
    fig = go.Figure()

    # Traces iniciales (θ = 0)
    init = build_data(0)
    labels = ['x (entrada)', 'y = g(x)', 'z = f(y)']

    # Traces 0–2: contornos de engranajes
    for i, key in enumerate(keys):
        line_c, fill_c, _ = COLORS[key]
        fig.add_trace(go.Scatter(
            x=init[i][0], y=init[i][1], fill='toself',
            fillcolor=fill_c, line=dict(color=line_c, width=1.5),
            name=labels[i], hoverinfo='skip'))

    # Traces 3–5: marcadores radiales
    for i, key in enumerate(keys):
        fig.add_trace(go.Scatter(
            x=init[3 + i][0], y=init[3 + i][1], mode='lines',
            line=dict(color=COLORS[key][0], width=4),
            showlegend=False, hoverinfo='skip'))

    # Traces 6–8: hubs
    for i, key in enumerate(keys):
        fig.add_trace(go.Scatter(
            x=init[6 + i][0], y=init[6 + i][1], fill='toself',
            fillcolor=COLORS[key][2],
            line=dict(color=COLORS[key][0], width=1),
            showlegend=False, hoverinfo='skip'))

    # ── Frames de animación ──
    n_steps = 100
    max_turns = 2.0
    thetas = np.linspace(0, max_turns * 2 * np.pi, n_steps)

    frames = []
    for idx, theta in enumerate(thetas):
        d = build_data(theta)
        frame_traces = []
        for i in range(9):
            frame_traces.append(go.Scatter(x=d[i][0], y=d[i][1]))
        frames.append(go.Frame(
            data=frame_traces,
            name=str(idx),
            layout=go.Layout(annotations=build_annotations(theta))
        ))
    fig.frames = frames

    # ── Slider ──
    slider_steps = []
    for idx, theta in enumerate(thetas):
        vx = theta / (2 * np.pi)
        slider_steps.append(dict(
            args=[[str(idx)], dict(
                mode="immediate",
                frame=dict(duration=0, redraw=True),
                transition=dict(duration=0))],
            label=f"{vx:.2f}",
            method="animate"))

    # ── Botones Play / Pausa ──
    updatemenus = [dict(
        type="buttons", showactive=False,
        y=-0.12, x=0.08, xanchor="right",
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(
                     frame=dict(duration=80, redraw=True),
                     fromcurrent=True,
                     transition=dict(duration=0))]),
            dict(label="⏸ Pausa", method="animate",
                 args=[[None], dict(
                     frame=dict(duration=0, redraw=True),
                     mode="immediate",
                     transition=dict(duration=0))])])]

    # ── Layout ──
    fig.update_layout(
        title=dict(
            text="<b><i>z</i> = (<i>f</i> ∘ <i>g</i>)(<i>x</i>)</b> — La regla de la cadena como engranajes",
            font=dict(size=18), x=0.5),
        sliders=[dict(
            active=0,
            currentvalue=dict(
                prefix="Rotación de x: ", suffix=" vueltas",
                font=dict(size=13)),
            pad=dict(t=60), steps=slider_steps,
            len=0.82, x=0.12)],
        updatemenus=updatemenus,
        xaxis=dict(
            scaleanchor="y", scaleratio=1,
            showgrid=False, zeroline=False, visible=False,
            range=[-2.5, cx3 + r3 + 1.5]),
        yaxis=dict(
            showgrid=False, zeroline=False, visible=False,
            range=[-r3 - 5.0, r3 + 1.5]),
        width=950, height=750,
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(x=0.82, y=1.0, font=dict(size=12)),
        annotations=build_annotations(0),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════
# Ejecución directa
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fig = crear_visualizacion()
    fig.show()
