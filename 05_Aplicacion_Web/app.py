"""
Aplicación web — Sistema Inteligente de Detección de Fraude en Tarjetas de Crédito
Trabajo Final de Minería de Datos (UCC - Campus Pasto)

Funcionalidades:
  1. Dashboard de fraude
  2. Alertas en tiempo real (simulación de un flujo de transacciones)
  3. Visualización de transacciones sospechosas
  4. Analizar una transacción o un archivo CSV
  5. Desempeño de los modelos (Precision, Recall, F1, ROC-AUC, matriz de confusión)

Ejecutar desde la carpeta del proyecto:
    streamlit run 05_Aplicacion_Web/app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RUTA_MODELOS = Path(__file__).resolve().parent.parent / "03_Modelos_Entrenados"
COLOR_NORMAL, COLOR_FRAUDE, COLOR_ALERTA = "#2E86AB", "#D62828", "#F4A261"

st.set_page_config(page_title="Detector de Fraude | Minería de Datos", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.titulo {font-size: 2.1rem; font-weight: 800; margin-bottom: 0;}
.subtitulo {font-size: 1.02rem; opacity: .75; margin-bottom: 1rem;}
.alerta {border-left: 6px solid #D62828; background: rgba(214,40,40,.08);
         padding: .55rem .9rem; border-radius: 6px; margin-bottom: .4rem;}
.alerta.media {border-left-color: #F4A261; background: rgba(244,162,97,.10);}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carga de modelos y datos (en caché)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos entrenados...")
def cargar_modelos():
    with open(RUTA_MODELOS / "metadata_modelos.json", encoding="utf-8") as f:
        meta = json.load(f)
    modelos = {n: joblib.load(RUTA_MODELOS / info["archivo"]) for n, info in meta["modelos"].items()}
    return meta, modelos


@st.cache_data(show_spinner="Cargando transacciones...")
def cargar_transacciones():
    return pd.read_parquet(RUTA_MODELOS / "transacciones_prueba.parquet")


@st.cache_data(show_spinner="Calculando probabilidades de fraude...")
def puntuar(nombre_modelo: str) -> np.ndarray:
    _, modelos = cargar_modelos()
    datos = cargar_transacciones()
    return modelos[nombre_modelo].predict_proba(datos[VARIABLES])[:, 1]


if not (RUTA_MODELOS / "metadata_modelos.json").exists():
    st.error("No se encontraron los modelos. Ejecute primero el notebook "
             "`02_Notebook_Analisis/Deteccion_Fraude_Tarjetas_Credito.ipynb`.")
    st.stop()

META, MODELOS = cargar_modelos()
VARIABLES = META["variables"]
TX = cargar_transacciones()


def nivel_riesgo(p: float, umbral: float) -> str:
    if p >= max(0.9, umbral):
        return "🔴 Crítico"
    if p >= umbral:
        return "🟠 Alto"
    if p >= umbral / 2:
        return "🟡 Medio"
    return "🟢 Bajo"


def formato_hora(h) -> str:
    return f"{int(h):02d}:00"


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ Detector de Fraude")
    st.caption("Trabajo Final · Minería de Datos · UCC Pasto")
    pagina = st.radio("Navegación", [
        "📊 Dashboard de fraude",
        "🚨 Alertas en tiempo real",
        "🔎 Transacciones sospechosas",
        "🧪 Analizar transacción",
        "📈 Desempeño de los modelos",
    ])
    st.divider()
    nombres_modelos = list(MODELOS.keys())
    modelo_sel = st.selectbox("Modelo de detección", nombres_modelos,
                              index=nombres_modelos.index(META["modelo_final"]),
                              help="Todos se entrenaron con SMOTE. El marcado como final fue el mejor en validación.")
    umbral_opt = META["modelos"][modelo_sel]["umbral_optimo"]
    umbral = st.slider("Umbral de alerta (probabilidad)", 0.005, 0.995, float(round(umbral_opt, 3)), 0.001, format="%.3f",
                       help=f"Umbral óptimo (máximo F1 con Recall ≥ 80 % en validación cruzada) para este modelo: {umbral_opt:.3f}")
    if modelo_sel == META["modelo_final"]:
        st.success(f"🏆 Modelo final seleccionado\n\nUmbral óptimo: **{umbral_opt:.3f}**")
    else:
        st.info(f"Umbral óptimo de este modelo: **{umbral_opt:.3f}**")
    st.divider()
    st.caption("Integrantes: Cristian Gomes · Ricardo Arias · Mateo Guerrero · Kevin Agreda")
    st.caption("Docente: PhD. Cristian Camilo Ordoñez Quintero")

PROB = puntuar(modelo_sel)
DATOS = TX.assign(prob_fraude=PROB, alerta=(PROB >= umbral).astype(int))
DATOS["riesgo"] = [nivel_riesgo(p, umbral) for p in PROB]
DATOS["hora_txt"] = DATOS["Hora"].map(formato_hora)

st.markdown('<p class="titulo">🛡️ Sistema Inteligente de Detección de Fraude</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitulo">Modelo activo: <b>{modelo_sel}</b> · Umbral de alerta: <b>{umbral:.3f}</b> · '
            f'Transacciones monitoreadas: <b>{len(DATOS):,}</b> (conjunto de prueba, nunca visto en entrenamiento)</p>',
            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 1. DASHBOARD
# ---------------------------------------------------------------------------
def pagina_dashboard():
    y, a = DATOS["Class"], DATOS["alerta"]
    tp = int(((a == 1) & (y == 1)).sum()); fp = int(((a == 1) & (y == 0)).sum())
    fn = int(((a == 0) & (y == 1)).sum())
    monto_detectado = DATOS.loc[(a == 1) & (y == 1), "Amount"].sum()
    monto_fraude = DATOS.loc[y == 1, "Amount"].sum()

    c = st.columns(5)
    c[0].metric("Transacciones", f"{len(DATOS):,}")
    c[1].metric("🚨 Alertas generadas", f"{int(a.sum()):,}", f"{a.mean() * 100:.3f}% del total", delta_color="off")
    c[2].metric("✅ Fraudes detectados", f"{tp} / {tp + fn}", f"Recall {tp / max(tp + fn, 1) * 100:.1f}%")
    c[3].metric("⚠️ Falsas alarmas", f"{fp:,}", f"Precision {tp / max(tp + fp, 1) * 100:.1f}%", delta_color="off")
    c[4].metric("💰 Dinero protegido", f"€ {monto_detectado:,.0f}",
                f"{monto_detectado / max(monto_fraude, 1) * 100:.1f}% del fraude", delta_color="off")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de la probabilidad de fraude")
        fig = px.histogram(DATOS, x="prob_fraude", color=DATOS["Class"].map({0: "Legítima", 1: "Fraude"}),
                           nbins=50, log_y=True, barmode="overlay", opacity=.75,
                           color_discrete_map={"Legítima": COLOR_NORMAL, "Fraude": COLOR_FRAUDE},
                           labels={"prob_fraude": "Probabilidad de fraude", "color": "Clase real"})
        fig.add_vline(x=umbral, line_dash="dash", annotation_text=f"Umbral {umbral:.3f}")
        fig.update_layout(height=370, margin=dict(t=10), yaxis_title="Transacciones (log)")
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.subheader("Resultado de las alertas")
        cm = confusion_matrix(y, a)
        etiquetas = [["Legítima correcta (TN)", "Falsa alarma (FP)"], ["Fraude no detectado (FN)", "Fraude detectado (TP)"]]
        texto = [[f"{etiquetas[i][j]}<br><b>{cm[i, j]:,}</b>" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(z=np.log1p(cm), x=["Pred: Legítima", "Pred: Fraude"], y=["Real: Legítima", "Real: Fraude"],
                                   text=texto, texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_layout(height=370, margin=dict(t=10), yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Alertas y fraudes por hora del día")
        por_hora = DATOS.groupby("Hora").agg(alertas=("alerta", "sum"), fraudes=("Class", "sum")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=por_hora["Hora"], y=por_hora["alertas"], name="Alertas", marker_color=COLOR_ALERTA)
        fig.add_scatter(x=por_hora["Hora"], y=por_hora["fraudes"], name="Fraudes reales", mode="lines+markers",
                        line=dict(color=COLOR_FRAUDE, width=3))
        fig.update_layout(height=360, margin=dict(t=10), xaxis_title="Hora", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
    with col4:
        st.subheader("Tasa histórica de fraude por hora (dataset completo)")
        h = pd.DataFrame(META["fraude_por_hora"])
        fig = px.area(h, x="Hora", y="tasa_fraude_%", color_discrete_sequence=[COLOR_FRAUDE],
                      labels={"tasa_fraude_%": "Tasa de fraude (%)"})
        fig.update_layout(height=360, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Monto vs. probabilidad de fraude de las transacciones con riesgo")
    riesgo = DATOS[DATOS["prob_fraude"] >= umbral / 2]
    fig = px.scatter(riesgo, x="Amount", y="prob_fraude", color="riesgo", hover_data=["id_transaccion", "hora_txt"],
                     color_discrete_map={"🔴 Crítico": COLOR_FRAUDE, "🟠 Alto": COLOR_ALERTA, "🟡 Medio": "#E9C46A"},
                     labels={"Amount": "Monto (€)", "prob_fraude": "Probabilidad de fraude", "riesgo": "Riesgo"})
    fig.add_hline(y=umbral, line_dash="dash")
    fig.update_layout(height=380, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# 2. ALERTAS EN TIEMPO REAL
# ---------------------------------------------------------------------------
def reiniciar_flujo(tasa_fraude: float):
    rng = np.random.default_rng()
    fraudes = TX.index[TX["Class"] == 1].to_numpy()
    legitimas = TX.index[TX["Class"] == 0].to_numpy()
    n = 2000
    n_fraude = int(round(n * tasa_fraude))
    indices = np.concatenate([rng.choice(fraudes, min(n_fraude, len(fraudes)), replace=False),
                              rng.choice(legitimas, n - min(n_fraude, len(fraudes)), replace=False)])
    rng.shuffle(indices)
    st.session_state.flujo = indices
    st.session_state.posicion = 0
    st.session_state.historial = []
    st.session_state.ejecutando = False


def pagina_tiempo_real():
    st.subheader("🚨 Monitoreo de transacciones en tiempo real")
    st.caption("Se simula un flujo continuo con transacciones reales del conjunto de prueba. "
               "Cada transacción se evalúa **en vivo** con el modelo seleccionado y, si supera el umbral, se genera una alerta.")

    c1, c2, c3 = st.columns([2, 2, 3])
    tasa = c1.slider("Proporción de fraudes en la simulación (%)", 0.2, 10.0, 3.0, 0.2,
                     help="En la realidad es 0,17%; se aumenta para que la demostración muestre alertas con frecuencia.") / 100
    velocidad = c2.slider("Transacciones por segundo", 1, 20, 5)
    with c3:
        st.write("")
        b1, b2, b3 = st.columns(3)
        if "flujo" not in st.session_state:
            reiniciar_flujo(tasa)
        if b1.button("▶️ Iniciar", width="stretch", type="primary"):
            st.session_state.ejecutando = True
        if b2.button("⏸️ Pausar", width="stretch"):
            st.session_state.ejecutando = False
        if b3.button("🔄 Reiniciar", width="stretch"):
            reiniciar_flujo(tasa)

    @st.fragment(run_every=1.0 if st.session_state.ejecutando else None)
    def panel_en_vivo():
        modelo = MODELOS[modelo_sel]
        if st.session_state.ejecutando and st.session_state.posicion < len(st.session_state.flujo):
            ini = st.session_state.posicion
            idx = st.session_state.flujo[ini: ini + velocidad]
            lote = TX.loc[idx]
            probs = modelo.predict_proba(lote[VARIABLES])[:, 1]  # predicción en vivo
            ahora = pd.Timestamp.now()
            for k, ((_, fila), p) in enumerate(zip(lote.iterrows(), probs)):
                registro = {"hora_llegada": (ahora + pd.Timedelta(milliseconds=k * 1000 // velocidad)).strftime("%H:%M:%S"),
                            "id_transaccion": fila["id_transaccion"], "monto": fila["Amount"], "hora_tx": formato_hora(fila["Hora"]),
                            "prob_fraude": float(p), "alerta": bool(p >= umbral), "fraude_real": int(fila["Class"]),
                            "riesgo": nivel_riesgo(p, umbral)}
                st.session_state.historial.append(registro)
                if registro["alerta"]:
                    st.toast(f"🚨 ALERTA {registro['id_transaccion']} · € {registro['monto']:,.2f} · "
                             f"prob. {registro['prob_fraude']:.1%}", icon="🚨")
            st.session_state.posicion += len(idx)
        elif st.session_state.posicion >= len(st.session_state.flujo):
            st.session_state.ejecutando = False

        hist = pd.DataFrame(st.session_state.historial)
        estado = "🟢 EN VIVO" if st.session_state.ejecutando else "⏸️ EN PAUSA"
        if hist.empty:
            st.info(f"{estado} — presione **▶️ Iniciar** para comenzar el monitoreo.")
            return
        alertas = hist[hist["alerta"]]
        m = st.columns(5)
        m[0].metric("Estado", estado)
        m[1].metric("Procesadas", f"{len(hist):,}")
        m[2].metric("🚨 Alertas", f"{len(alertas):,}")
        m[3].metric("Fraudes confirmados en alertas", f"{int(alertas['fraude_real'].sum()) if len(alertas) else 0}")
        m[4].metric("Monto bajo alerta", f"€ {alertas['monto'].sum():,.2f}" if len(alertas) else "€ 0")

        ult = hist.tail(150).reset_index(drop=True)
        fig = go.Figure()
        fig.add_scatter(y=ult["prob_fraude"], mode="lines", line=dict(color=COLOR_NORMAL, width=1.5), name="Probabilidad")
        al = ult[ult["alerta"]]
        fig.add_scatter(x=al.index, y=al["prob_fraude"], mode="markers", marker=dict(color=COLOR_FRAUDE, size=11, symbol="x"),
                        name="Alerta", text=al["id_transaccion"])
        fig.add_hline(y=umbral, line_dash="dash", line_color=COLOR_FRAUDE, annotation_text="Umbral")
        fig.update_layout(height=300, margin=dict(t=20, b=10), yaxis_range=[-0.02, 1.05],
                          xaxis_title="Últimas 150 transacciones", yaxis_title="Prob. fraude", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("#### 🔔 Últimas alertas")
            if alertas.empty:
                st.caption("Sin alertas por ahora.")
            for _, r in alertas.tail(8).iloc[::-1].iterrows():
                clase = "alerta" if r["prob_fraude"] >= 0.9 else "alerta media"
                verificacion = "✔ fraude confirmado" if r["fraude_real"] else "✖ falsa alarma"
                st.markdown(f'<div class="{clase}"><b>{r["riesgo"]}</b> · {r["hora_llegada"]} · <b>{r["id_transaccion"]}</b> · '
                            f'€ {r["monto"]:,.2f} · prob. <b>{r["prob_fraude"]:.1%}</b> <span style="opacity:.7">({verificacion})</span></div>',
                            unsafe_allow_html=True)
        with col_b:
            st.markdown("#### 📜 Flujo de transacciones")
            st.dataframe(hist.tail(12).iloc[::-1][["hora_llegada", "id_transaccion", "monto", "prob_fraude", "riesgo"]],
                         hide_index=True, width="stretch",
                         column_config={"monto": st.column_config.NumberColumn("Monto (€)", format="%.2f"),
                                        "prob_fraude": st.column_config.ProgressColumn("Prob. fraude", min_value=0, max_value=1, format="%.3f")})

    panel_en_vivo()


# ---------------------------------------------------------------------------
# 3. TRANSACCIONES SOSPECHOSAS
# ---------------------------------------------------------------------------
def pagina_sospechosas():
    st.subheader("🔎 Transacciones sospechosas")
    c1, c2, c3 = st.columns(3)
    prob_min = c1.slider("Probabilidad mínima", 0.0, 1.0, float(round(umbral, 3)), 0.001, format="%.3f")
    monto_max = float(DATOS["Amount"].max())
    rango_monto = c2.slider("Rango de monto (€)", 0.0, monto_max, (0.0, monto_max))
    rango_hora = c3.slider("Rango de hora del día", 0, 23, (0, 23))
    sosp = DATOS[(DATOS["prob_fraude"] >= prob_min) & DATOS["Amount"].between(*rango_monto) & DATOS["Hora"].between(*rango_hora)]
    sosp = sosp.sort_values("prob_fraude", ascending=False)

    m = st.columns(4)
    m[0].metric("Sospechosas", f"{len(sosp):,}")
    m[1].metric("Monto total", f"€ {sosp['Amount'].sum():,.2f}")
    m[2].metric("Fraudes confirmados", f"{int(sosp['Class'].sum())}")
    m[3].metric("Precisión de la lista", f"{sosp['Class'].mean() * 100:.1f}%" if len(sosp) else "—")

    vista = sosp[["id_transaccion", "hora_txt", "Amount", "prob_fraude", "riesgo", "Class"]].rename(columns={
        "id_transaccion": "ID", "hora_txt": "Hora", "Amount": "Monto (€)", "prob_fraude": "Prob. fraude",
        "riesgo": "Riesgo", "Class": "Fraude real"})
    vista["Fraude real"] = vista["Fraude real"].map({1: "✔ Sí", 0: "✖ No"})
    st.dataframe(vista, hide_index=True, width="stretch", height=360,
                 column_config={"Prob. fraude": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                                "Monto (€)": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("⬇️ Descargar reporte CSV", vista.to_csv(index=False).encode("utf-8"),
                       "transacciones_sospechosas.csv", "text/csv")

    if sosp.empty:
        return
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribución por nivel de riesgo")
        conteo = sosp["riesgo"].value_counts().reset_index()
        fig = px.pie(conteo, names="riesgo", values="count", hole=.45,
                     color="riesgo", color_discrete_map={"🔴 Crítico": COLOR_FRAUDE, "🟠 Alto": COLOR_ALERTA,
                                                         "🟡 Medio": "#E9C46A", "🟢 Bajo": "#90BE6D"})
        fig.update_layout(height=340, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.markdown("#### Perfil de una transacción sospechosa")
        tx_id = st.selectbox("Seleccione una transacción", sosp["id_transaccion"].head(200))
        fila = DATOS.loc[DATOS["id_transaccion"] == tx_id].iloc[0]
        top = list(META["importancia_xgboost"].keys())[:10]
        top = [v for v in top if v.startswith("V")][:8]
        media_leg = TX.loc[TX["Class"] == 0, top].mean()
        media_fra = TX.loc[TX["Class"] == 1, top].mean()
        fig = go.Figure()
        fig.add_bar(x=top, y=media_leg, name="Promedio legítimas", marker_color=COLOR_NORMAL, opacity=.6)
        fig.add_bar(x=top, y=media_fra, name="Promedio fraudes", marker_color=COLOR_FRAUDE, opacity=.6)
        fig.add_scatter(x=top, y=fila[top].values.astype(float), name=tx_id, mode="markers+lines",
                        marker=dict(size=12, color="black"))
        fig.update_layout(height=340, margin=dict(t=10), barmode="group", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Monto € {fila['Amount']:,.2f} · Hora {fila['hora_txt']} · Probabilidad {fila['prob_fraude']:.1%} · "
                   f"{'Fraude real' if fila['Class'] else 'Legítima'}. Si la línea negra se parece al perfil rojo, el patrón es típico de fraude.")


# ---------------------------------------------------------------------------
# 4. ANALIZAR TRANSACCIÓN
# ---------------------------------------------------------------------------
def medidor(prob: float, titulo: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": titulo, "font": {"size": 14}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLOR_FRAUDE if prob >= umbral else COLOR_NORMAL},
               "steps": [{"range": [0, umbral * 100], "color": "rgba(46,134,171,.15)"},
                         {"range": [umbral * 100, 100], "color": "rgba(214,40,40,.15)"}],
               "threshold": {"line": {"color": "black", "width": 3}, "value": umbral * 100}}))
    fig.update_layout(height=230, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def pagina_analizar():
    st.subheader("🧪 Analizar transacciones")
    tab1, tab2 = st.tabs(["Una transacción", "Archivo CSV (lote)"])
    with tab1:
        st.caption("Escoja una transacción real del conjunto de prueba y modifique su monto u hora para ver cómo cambia el riesgo.")
        c1, c2 = st.columns(2)
        if "tx_analisis" not in st.session_state:
            st.session_state.tx_analisis = TX[TX["Class"] == 1].sample(1).index[0]
        if c1.button("🎲 Cargar un FRAUDE real al azar", width="stretch"):
            st.session_state.tx_analisis = TX[TX["Class"] == 1].sample(1).index[0]
        if c2.button("🎲 Cargar una transacción LEGÍTIMA al azar", width="stretch"):
            st.session_state.tx_analisis = TX[TX["Class"] == 0].sample(1).index[0]
        base = TX.loc[st.session_state.tx_analisis]
        c3, c4 = st.columns(2)
        monto = c3.number_input("Monto (€)", 0.0, 30000.0, float(base["Amount"]), 1.0)
        hora = c4.slider("Hora del día", 0, 23, int(base["Hora"]))
        with st.expander("Ver / editar componentes PCA V1–V28"):
            editadas = st.data_editor(base[[f"V{i}" for i in range(1, 29)]].astype(float).to_frame("valor").T,
                                      hide_index=True, width="stretch")
        entrada = editadas.copy()
        entrada["Amount"] = monto
        entrada["Hora"] = hora
        entrada = entrada[VARIABLES]
        st.markdown(f"**Transacción {base['id_transaccion']}** — clase real: "
                    f"{'🔴 FRAUDE' if base['Class'] == 1 else '🟢 Legítima'}")
        cols = st.columns(len(MODELOS))
        for col, (nombre, modelo) in zip(cols, MODELOS.items()):
            p = float(modelo.predict_proba(entrada)[:, 1][0])
            col.plotly_chart(medidor(p, nombre), width="stretch")
            col.markdown(f"<div style='text-align:center'><b>{nivel_riesgo(p, umbral)}</b> — "
                         f"{'🚨 GENERA ALERTA' if p >= umbral else '✅ Aprobada'}</div>", unsafe_allow_html=True)
    with tab2:
        st.caption(f"Suba un CSV con las columnas {', '.join(VARIABLES[:3])} … V28, Amount y Hora "
                   "(o Time en lugar de Hora, como en el dataset original de Kaggle).")
        archivo = st.file_uploader("Archivo CSV", type="csv")
        if archivo is not None:
            lote = pd.read_csv(archivo)
            if "Hora" not in lote.columns and "Time" in lote.columns:
                lote["Hora"] = (lote["Time"] // 3600) % 24
            faltan = [v for v in VARIABLES if v not in lote.columns]
            if faltan:
                st.error(f"Faltan columnas: {faltan}")
            else:
                lote["prob_fraude"] = MODELOS[modelo_sel].predict_proba(lote[VARIABLES])[:, 1]
                lote["alerta"] = np.where(lote["prob_fraude"] >= umbral, "🚨 Sí", "No")
                lote["riesgo"] = [nivel_riesgo(p, umbral) for p in lote["prob_fraude"]]
                st.success(f"{len(lote):,} transacciones analizadas · {int((lote['prob_fraude'] >= umbral).sum()):,} alertas")
                st.dataframe(lote.sort_values("prob_fraude", ascending=False), width="stretch", hide_index=True)
                st.download_button("⬇️ Descargar resultados", lote.to_csv(index=False).encode("utf-8"),
                                   "resultados_deteccion.csv", "text/csv")
        st.download_button("⬇️ Descargar CSV de ejemplo (50 transacciones)",
                           TX.sample(50, random_state=1)[VARIABLES + ["Class"]].to_csv(index=False).encode("utf-8"),
                           "ejemplo_transacciones.csv", "text/csv")


# ---------------------------------------------------------------------------
# 5. DESEMPEÑO DE LOS MODELOS
# ---------------------------------------------------------------------------
def pagina_desempeno():
    st.subheader("📈 Desempeño de los modelos (conjunto de prueba)")
    y = TX["Class"].values
    filas, curvas = [], {}
    for nombre in MODELOS:
        p = puntuar(nombre)
        u = umbral if nombre == modelo_sel else META["modelos"][nombre]["umbral_optimo"]
        pred = (p >= u).astype(int)
        filas.append({"Modelo": nombre, "Umbral": u, "Precision": precision_score(y, pred, zero_division=0),
                      "Recall": recall_score(y, pred), "F1": f1_score(y, pred),
                      "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p)})
        curvas[nombre] = p
    tabla = pd.DataFrame(filas).set_index("Modelo")
    st.dataframe(tabla.style.format("{:.4f}").background_gradient(cmap="Greens", subset=["F1", "ROC-AUC", "PR-AUC"]),
                 width="stretch")
    st.caption("El modelo activo usa el umbral de la barra lateral; los demás usan su umbral óptimo de validación cruzada.")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for nombre, p in curvas.items():
            fpr, tpr, _ = roc_curve(y, p)
            fig.add_scatter(x=fpr, y=tpr, name=f"{nombre} ({roc_auc_score(y, p):.4f})", mode="lines")
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Azar", line=dict(dash="dash", color="gray"))
        fig.update_layout(title="Curva ROC", xaxis_title="Tasa de falsos positivos", yaxis_title="Recall",
                          height=420, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = go.Figure()
        for nombre, p in curvas.items():
            prec, rec, _ = precision_recall_curve(y, p)
            fig.add_scatter(x=rec, y=prec, name=f"{nombre} ({average_precision_score(y, p):.4f})", mode="lines")
        fig.update_layout(title="Curva Precision–Recall", xaxis_title="Recall", yaxis_title="Precision",
                          height=420, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"#### Matriz de confusión — {modelo_sel}")
        cm = confusion_matrix(y, (curvas[modelo_sel] >= umbral).astype(int))
        fig = px.imshow(cm, text_auto=",", x=["Pred: Legítima", "Pred: Fraude"], y=["Real: Legítima", "Real: Fraude"],
                        color_continuous_scale="Blues", zmax=max(cm[1].sum(), cm[0, 1]) * 3)
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with col4:
        st.markdown("#### Variables más importantes (XGBoost)")
        imp = pd.Series(META["importancia_xgboost"]).sort_values()
        fig = px.bar(imp, orientation="h", color_discrete_sequence=[COLOR_FRAUDE], labels={"value": "Importancia", "index": ""})
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Validación cruzada estratificada (5 folds, entrenamiento + validación)")
    cv = {n: {m.upper(): f"{v['media']:.4f} ± {v['std']:.4f}" for m, v in r.items()} for n, r in META["validacion_cruzada"].items()}
    st.dataframe(pd.DataFrame(cv).T, width="stretch")


# ---------------------------------------------------------------------------
PAGINAS = {
    "📊 Dashboard de fraude": pagina_dashboard,
    "🚨 Alertas en tiempo real": pagina_tiempo_real,
    "🔎 Transacciones sospechosas": pagina_sospechosas,
    "🧪 Analizar transacción": pagina_analizar,
    "📈 Desempeño de los modelos": pagina_desempeno,
}
PAGINAS[pagina]()
