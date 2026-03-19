"""
red_neuronal_simple.py
══════════════════════════════════════════════════════════════════════════════
Simulación completa del entrenamiento de una red neuronal simple (2→1)
reproduciendo exactamente el proceso mostrado en la hoja de cálculo.

Compuerta AND, activación sigmoide, pérdida MSE, Descenso por Gradiente.
══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# 1. FUNCIONES DE LA RED NEURONAL
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(z):
    """Función de activación sigmoide: σ(z) = 1 / (1 + e^{-z})"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_deriv(z):
    """Derivada de la sigmoide: σ'(z) = σ(z) · (1 − σ(z))"""
    s = sigmoid(z)
    return s * (1.0 - s)


def forward_pass(X, b, w1, w2):
    """
    Propagación hacia adelante.
    z = b + w1*x1 + w2*x2
    a = σ(z)
    """
    z = b + w1 * X[:, 0] + w2 * X[:, 1]
    a = sigmoid(z)
    return z, a


def compute_cost(a, y):
    """Costo total: C = Σ (a_i - y_i)²"""
    return float(np.sum((a - y) ** 2))


def compute_gradients(X, y, z, a):
    """
    Calcula los gradientes usando la regla de la cadena.

    ∂C/∂w_i = mean[ 2(a-y) · σ'(z) · x_i ]
    ∂C/∂b   = mean[ 2(a-y) · σ'(z) · 1   ]

    Retorna: grad_b, grad_w1, grad_w2
    """
    dC_da = 2.0 * (a - y)
    da_dz = sigmoid_deriv(z)
    delta = dC_da * da_dz

    grad_b  = float(np.mean(delta))
    grad_w1 = float(np.mean(delta * X[:, 0]))
    grad_w2 = float(np.mean(delta * X[:, 1]))

    return grad_b, grad_w1, grad_w2


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATOS Y ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train(b0=1.0, w1_0=-2.0, w2_0=3.0, lr=10, n_steps=50, verbose=True):
    """
    Entrena la red neuronal para aprender la compuerta AND.

    Parámetros
    ----------
    b0, w1_0, w2_0 : valores iniciales (iguales a los de la hoja de cálculo)
    lr             : tasa de aprendizaje (learning rate k)
    n_steps        : número de pasos de gradiente
    verbose        : imprimir tabla de progreso

    Retorna
    -------
    history : lista de dicts con el estado en cada paso
    """
    # Datos: AND gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1], dtype=float)

    b, w1, w2 = b0, w1_0, w2_0
    history = []

    if verbose:
        print("\n" + "═" * 62)
        print("  ENTRENAMIENTO — Red Neuronal Simple (AND Gate)")
        print("  b₀={:.1f}  w₁₀={:.1f}  w₂₀={:.1f}  k={}  pasos={}".format(
            b0, w1_0, w2_0, lr, n_steps))
        print("═" * 62)
        print(f"{'Paso':>5} │ {'b':>8} │ {'w₁':>8} │ {'w₂':>8} │ {'Costo':>8}")
        print("─" * 50)

    for step in range(n_steps + 1):
        z, a = forward_pass(X, b, w1, w2)
        cost = compute_cost(a, y)

        # Calcular gradientes por ejemplo (para el detalle de la tabla)
        per_example = []
        for i, (xi, yi) in enumerate(zip(X, y)):
            zi  = b + w1 * xi[0] + w2 * xi[1]
            ai  = sigmoid(zi)
            ci  = (ai - yi) ** 2
            dca = 2 * (ai - yi)
            daz = sigmoid_deriv(zi)
            delta_i = dca * daz
            per_example.append({
                'x1': xi[0], 'x2': xi[1],
                'z': zi, 'a': ai, 'C': ci,
                'dC_da': dca, 'da_dz': daz,
                'db': delta_i * 1,
                'dw1': delta_i * xi[0],
                'dw2': delta_i * xi[1],
            })

        history.append({
            'step': step, 'b': b, 'w1': w1, 'w2': w2,
            'cost': cost, 'outputs': a.copy(),
            'per_example': per_example,
        })

        if verbose and (step <= 15 or step == n_steps):
            print(f"{step:>5} │ {b:>8.3f} │ {w1:>8.3f} │ {w2:>8.3f} │ {cost:>8.4f}")

        if step == n_steps:
            break

        # Actualizar pesos
        grad_b, grad_w1, grad_w2 = compute_gradients(X, y, z, a)
        b  -= lr * grad_b
        w1 -= lr * grad_w1
        w2 -= lr * grad_w2

    if verbose:
        print("─" * 50)
        print(f"\n✅ Entrenamiento completado.")
        print(f"   Costo inicial : {history[0]['cost']:.4f}")
        print(f"   Costo final   : {history[-1]['cost']:.6f}")
        print(f"   Reducción     : {(1-history[-1]['cost']/history[0]['cost'])*100:.1f}%")

        print("\n📋 Predicciones finales:")
        b_f, w1_f, w2_f = history[-1]['b'], history[-1]['w1'], history[-1]['w2']
        _, a_f = forward_pass(X, b_f, w1_f, w2_f)
        print(f"  {'x1':>3} {'x2':>3} │ {'y':>5} │ {'ŷ (prob)':>10} │ {'clase':>6} │")
        print("  " + "─" * 40)
        for xi, yi, ai in zip(X, y, a_f):
            clase = int(ai >= 0.5)
            ok = "✅" if clase == int(yi) else "❌"
            print(f"  {int(xi[0]):>3} {int(xi[1]):>3} │ {int(yi):>5} │ {ai:>10.4f} │ {clase:>6} │ {ok}")

    return history, X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3. VISUALIZACIONES PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

def plot_cost_curve(history):
    """Gráfico de la evolución del costo durante el entrenamiento."""
    costs = [h['cost'] for h in history]
    steps = [h['step'] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=costs,
        mode='lines+markers',
        line=dict(color='#E74C3C', width=2.5),
        marker=dict(size=5),
        hovertemplate='Paso %{x}<br>Costo: %{y:.4f}<extra></extra>'
    ))
    fig.add_annotation(x=0, y=costs[0],
        text=f'Inicio C={costs[0]:.3f}',
        showarrow=True, arrowhead=2, ax=50, ay=-30,
        font=dict(color='#E74C3C', size=12))
    fig.add_annotation(x=steps[-1], y=costs[-1],
        text=f'Final C={costs[-1]:.4f}',
        showarrow=True, arrowhead=2, ax=-50, ay=-30,
        font=dict(color='#27AE60', size=12))
    fig.update_layout(
        title='📉 Curva de Aprendizaje — Evolución del Costo',
        xaxis_title='Paso', yaxis_title='Costo C',
        template='plotly_white', height=400
    )
    return fig


def plot_weights_evolution(history):
    """Evolución de los 3 parámetros (b, w1, w2) durante el entrenamiento."""
    steps  = [h['step'] for h in history]
    b_h    = [h['b']    for h in history]
    w1_h   = [h['w1']   for h in history]
    w2_h   = [h['w2']   for h in history]

    fig = make_subplots(rows=1, cols=3,
        subplot_titles=('Sesgo b', 'Peso w₁', 'Peso w₂'))

    for col, (vals, label, color) in enumerate(
            zip([b_h, w1_h, w2_h], ['b', 'w₁', 'w₂'],
                ['#9B59B6', '#3498DB', '#E67E22']), 1):
        fig.add_trace(go.Scatter(
            x=steps, y=vals, mode='lines+markers',
            name=label, line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate=f'Paso %{{x}}<br>{label}=%{{y:.4f}}<extra></extra>'
        ), row=1, col=col)
        fig.add_hline(y=vals[-1], line_dash='dash',
                      line_color=color, opacity=0.4, row=1, col=col)

    fig.update_layout(
        title='⚙️ Evolución de los Parámetros durante el Entrenamiento',
        template='plotly_white', height=380, showlegend=False
    )
    return fig


def plot_contour_with_path(history, X, y):
    """
    Mapa de contorno de la función de costo C(w1, w2)
    con la trayectoria real de los pesos y flechas de gradiente.
    """
    b_mean = np.mean([h['b'] for h in history])
    w1_h = [h['w1'] for h in history]
    w2_h = [h['w2'] for h in history]

    # Malla
    w1_r = np.linspace(min(w1_h) - 1, max(w1_h) + 1, 150)
    w2_r = np.linspace(min(w2_h) - 0.5, max(w2_h) + 0.5, 150)
    W1g, W2g = np.meshgrid(w1_r, w2_r)
    Cg = np.zeros_like(W1g)
    for i in range(W1g.shape[0]):
        for j in range(W1g.shape[1]):
            _, a_tmp = forward_pass(X, b_mean, W1g[i,j], W2g[i,j])
            Cg[i,j] = compute_cost(a_tmp, y)

    costs = [h['cost'] for h in history]

    fig = go.Figure()

    # Contorno
    fig.add_trace(go.Contour(
        x=w1_r, y=w2_r, z=Cg,
        colorscale='RdYlGn_r',
        contours=dict(showlabels=True, labelfont=dict(size=9)),
        colorbar=dict(title='Costo'),
        hovertemplate='w₁=%{x:.2f}<br>w₂=%{y:.2f}<br>C=%{z:.4f}<extra></extra>'
    ))

    # Trayectoria
    fig.add_trace(go.Scatter(
        x=w1_h, y=w2_h,
        mode='lines+markers',
        line=dict(color='white', width=2.5),
        marker=dict(size=6, color=costs, colorscale='Blues', showscale=False),
        name='Trayectoria',
        text=[str(h['step']) for h in history],
        hovertemplate='Paso %{text}<br>w₁=%{x:.3f}<br>w₂=%{y:.3f}<extra></extra>'
    ))

    # Inicio / fin
    fig.add_trace(go.Scatter(x=[w1_h[0]], y=[w2_h[0]],
        mode='markers+text', marker=dict(size=14, color='red', symbol='star'),
        text=['Inicio'], textposition='top right',
        textfont=dict(color='white', size=12), name='Inicio'))
    fig.add_trace(go.Scatter(x=[w1_h[-1]], y=[w2_h[-1]],
        mode='markers+text', marker=dict(size=14, color='lime', symbol='star'),
        text=['Mínimo'], textposition='top right',
        textfont=dict(color='white', size=12), name='Mínimo'))

    # Flechas de gradiente cada 5 pasos
    for idx in range(0, len(history)-1, 5):
        h = history[idx]
        z_tmp, a_tmp = forward_pass(X, h['b'], h['w1'], h['w2'])
        _, gw1, gw2 = compute_gradients(X, y, z_tmp, a_tmp)
        sc = 0.35
        fig.add_annotation(
            x=h['w1'] - sc*gw1, y=h['w2'] - sc*gw2,
            ax=h['w1'], ay=h['w2'],
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=3,
            arrowsize=1.3, arrowwidth=1.5, arrowcolor='cyan'
        )

    fig.update_layout(
        title='🗺️ Mapa de Contorno y Descenso por Gradiente',
        xaxis_title='w₁', yaxis_title='w₂',
        template='plotly_dark', height=600,
        legend=dict(bgcolor='rgba(0,0,0,0.5)', font=dict(color='white'))
    )
    return fig


def plot_3d_trajectory(history, X, y):
    """Trayectoria 3D en el espacio (w1, w2, Costo)."""
    w1_h  = [h['w1']  for h in history]
    w2_h  = [h['w2']  for h in history]
    costs = [h['cost'] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=w1_h, y=w2_h, z=costs,
        mode='lines+markers',
        line=dict(color=costs, colorscale='Viridis', width=5),
        marker=dict(size=4, color=costs, colorscale='Viridis',
                    showscale=True, colorbar=dict(title='Costo')),
        hovertemplate='w₁=%{x:.3f}<br>w₂=%{y:.3f}<br>C=%{z:.4f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter3d(
        x=[w1_h[0]], y=[w2_h[0]], z=[costs[0]],
        mode='markers+text',
        marker=dict(size=10, color='red'),
        text=['Inicio'], textposition='top center', name='Inicio'
    ))
    fig.add_trace(go.Scatter3d(
        x=[w1_h[-1]], y=[w2_h[-1]], z=[costs[-1]],
        mode='markers+text',
        marker=dict(size=10, color='lime', symbol='diamond'),
        text=['Final'], textposition='top center', name='Final'
    ))
    fig.update_layout(
        title='🏔️ Trayectoria 3D en el Espacio de Pesos',
        scene=dict(xaxis_title='w₁', yaxis_title='w₂', zaxis_title='Costo',
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
        height=550, template='plotly_white'
    )
    return fig


def plot_learned_surface(history, X, y):
    """Superficie de activación aprendida por la red."""
    bf   = history[-1]['b']
    w1f  = history[-1]['w1']
    w2f  = history[-1]['w2']

    xg = np.linspace(-0.15, 1.15, 80)
    yg = np.linspace(-0.15, 1.15, 80)
    X1g, X2g = np.meshgrid(xg, yg)
    Xflat = np.column_stack([X1g.ravel(), X2g.ravel()])
    _, Aflat = forward_pass(Xflat, bf, w1f, w2f)
    Agrid = Aflat.reshape(X1g.shape)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xg, y=yg, z=Agrid,
        colorscale='Blues', opacity=0.85,
        showscale=True, colorbar=dict(title='ŷ')
    ))
    colors_pts = ['red' if yi == 0 else 'lime' for yi in y]
    for i, (xi, yi) in enumerate(zip(X, y)):
        fig.add_trace(go.Scatter3d(
            x=[xi[0]], y=[xi[1]], z=[float(yi)],
            mode='markers+text',
            marker=dict(size=9, color=colors_pts[i],
                        line=dict(color='black', width=1)),
            text=[f'({int(xi[0])},{int(xi[1])})→{int(yi)}'],
            textposition='top center', showlegend=False
        ))
    fig.update_layout(
        title='🧠 Función Aprendida: (x₁, x₂) → ŷ',
        scene=dict(xaxis_title='x₁', yaxis_title='x₂', zaxis_title='ŷ',
                   camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))),
        height=550, template='plotly_white'
    )
    return fig


def plot_detail_step(history, step_idx, X, y):
    """
    Tabla detallada de un paso específico del entrenamiento,
    similar a la hoja de cálculo de las imágenes.
    """
    h = history[step_idx]
    print(f"\n{'═'*80}")
    print(f"  DETALLE — Paso {h['step']}   |   b={h['b']:.3f}   w₁={h['w1']:.3f}   w₂={h['w2']:.3f}")
    print(f"{'═'*80}")

    header = (f"{'x1':>4} {'x2':>4} │ {'z':>7} {'a':>7} {'C':>7} │"
              f" {'∂C/∂a':>8} {'∂a/∂z':>8} │"
              f" {'∂b':>7} {'∂w1':>7} {'∂w2':>7} │"
              f" {'δb':>7} {'δw1':>7} {'δw2':>7}")
    print(header)
    print("─" * len(header))

    sum_db = sum_dw1 = sum_dw2 = 0.0
    N = len(h['per_example'])
    for pe in h['per_example']:
        print(f"{int(pe['x1']):>4} {int(pe['x2']):>4} │"
              f" {pe['z']:>7.3f} {pe['a']:>7.3f} {pe['C']:>7.3f} │"
              f" {pe['dC_da']:>8.3f} {pe['da_dz']:>8.3f} │"
              f" {pe['db']:>7.3f} {pe['dw1']:>7.3f} {pe['dw2']:>7.3f} │"
              f" {pe['db']/N:>7.3f} {pe['dw1']/N:>7.3f} {pe['dw2']/N:>7.3f}")
        sum_db  += pe['db']  / N
        sum_dw1 += pe['dw1'] / N
        sum_dw2 += pe['dw2'] / N

    total_cost = h['cost']
    print(f"{'':>44} Total C: {total_cost:>7.3f}")
    print(f"{'':>72} Σδ: {sum_db:>7.3f} {sum_dw1:>7.3f} {sum_dw2:>7.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Entrenar — mismos parámetros que la hoja de cálculo
    history, X, y = train(b0=1.0, w1_0=-2.0, w2_0=3.0, lr=10, n_steps=50)

    # Mostrar detalle del paso 0 y paso 5
    plot_detail_step(history, 0, X, y)
    plot_detail_step(history, 5, X, y)

    # Generar todas las visualizaciones
    figs = {
        'curva_aprendizaje'  : plot_cost_curve(history),
        'evolucion_pesos'    : plot_weights_evolution(history),
        'mapa_contorno'      : plot_contour_with_path(history, X, y),
        'trayectoria_3d'     : plot_3d_trajectory(history, X, y),
        'superficie_aprendida': plot_learned_surface(history, X, y),
    }

    # Guardar HTMLs interactivos
    for nombre, fig in figs.items():
        path = f"/home/claude/{nombre}.html"
        fig.write_html(path, include_plotlyjs='cdn')
        print(f"  💾 {nombre}.html guardado.")

    print("\n✅ Todos los gráficos generados exitosamente.")
