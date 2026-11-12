# 🛡️ Sistema Inteligente de Detección de Transacciones Fraudulentas

**Trabajo Final — Minería de Datos** · Universidad Cooperativa de Colombia, Campus Pasto
**Docente:** PhD. Cristian Camilo Ordoñez Quintero
**Integrantes:** Cristian Gomes · Ricardo Arias · Mateo Guerrero · Kevin Agreda

Sistema de *machine learning* y aprendizaje profundo que detecta fraude en tarjetas de crédito con el dataset
[Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Incluye una aplicación web con dashboard, alertas en tiempo real y visualización de transacciones sospechosas.

---

## 📁 Estructura del proyecto

```
trabajo final mineria de datos/
├── 01_Datos/
│   └── creditcard.csv                         ← dataset de Kaggle (284.807 transacciones, 150 MB)
├── 02_Notebook_Analisis/
│   └── Deteccion_Fraude_Tarjetas_Credito.ipynb ← TODO el análisis, con las preguntas respondidas
├── 03_Modelos_Entrenados/
│   ├── modelo_xgboost_smote.joblib            ← 🏆 modelo final
│   ├── modelo_random_forest_smote.joblib
│   ├── modelo_red_neuronal_mlp_smote.joblib
│   ├── metadata_modelos.json                  ← umbrales, métricas, validación cruzada
│   ├── metricas_umbral_05.csv / metricas_umbral_optimo.csv
│   └── transacciones_prueba.parquet           ← conjunto de prueba que usa la app
├── 04_Graficos_Resultados/                    ← 12 gráficos generados por el notebook
├── 05_Aplicacion_Web/
│   └── app.py                                 ← aplicación Streamlit
├── 06_Informe_Final/
│   ├── Informe_Final_Deteccion_Fraude.md      ← informe con resultados y conclusiones
│   └── Deteccion_Fraude_Tarjetas_Credito.html ← notebook ejecutado, para abrir en el navegador
├── requirements.txt
└── README.md
```

## ✅ Cumplimiento de los requerimientos del docente

| Requerimiento | Implementación |
|---|---|
| Balanceo de clases (**SMOTE**) | `imblearn` SMOTE dentro de un Pipeline, aplicado **solo al entrenamiento** (sin fuga de datos) |
| **Random Forest** | `RandomForestClassifier` (150 árboles) con y sin SMOTE |
| **XGBoost** (Boosting) | `XGBClassifier` (400 árboles, *hist*) con y sin SMOTE → **modelo final** |
| Aprendizaje profundo | Red neuronal **MLP** de 3 capas ocultas (64-32-16, ReLU) |
| **Precision, Recall, F1, ROC-AUC, matriz de confusión** | Partes 6 y 7 del notebook, más PR-AUC, curvas ROC/PR y validación cruzada de 5 folds |
| **Dashboard de fraude** | Página "📊 Dashboard de fraude" de la app |
| **Alertas en tiempo real** | Página "🚨 Alertas en tiempo real": flujo simulado puntuado en vivo, con notificaciones |
| **Visualización de transacciones sospechosas** | Página "🔎 Transacciones sospechosas": filtros, nivel de riesgo, perfil y descarga en CSV |
| Extra | Análisis de transacción individual o por lote (CSV) y página de desempeño de los modelos |

## 📊 Resultados (conjunto de prueba: 56.746 transacciones nunca vistas, 95 fraudes)

| Modelo | Umbral | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **🏆 XGBoost + SMOTE** | 0,968 | **0,985** | 0,695 | **0,815** | 0,967 | 0,808 |
| Random Forest + SMOTE | 0,617 | 0,886 | 0,737 | 0,805 | **0,974** | 0,809 |
| Red Neuronal MLP + SMOTE | 0,214 | 0,744 | 0,705 | 0,724 | 0,924 | 0,732 |
| Regresión Logística + SMOTE (base) | — | 0,794 | 0,768 | 0,781 | 0,955 | 0,676 |

El modelo final detecta 66 de 95 fraudes con **una sola falsa alarma** entre 56.651 transacciones legítimas y protege ≈ 60 % del dinero en fraude.
El umbral se eligió con validación cruzada *out-of-fold* (máximo F1 con Recall ≥ 80 %) y se puede ajustar en vivo desde la aplicación.

## 🚀 Cómo ejecutarlo

1. Instalar las dependencias (Python 3.10 o superior):
   ```bash
   pip install -r requirements.txt
   ```
2. (Solo si no está) descargar `creditcard.csv` de [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) y guardarlo en `01_Datos/`.
3. (Opcional, los modelos ya están entrenados) volver a ejecutar el notebook: `02_Notebook_Analisis/Deteccion_Fraude_Tarjetas_Credito.ipynb` (≈ 15 min).
4. Abrir la aplicación web desde esta carpeta:
   ```bash
   streamlit run 05_Aplicacion_Web/app.py
   ```
   Se abre en `http://localhost:8501`.

> Nota para GitHub: `creditcard.csv` pesa 150 MB (GitHub admite máximo 100 MB por archivo), por eso está en `.gitignore`.
