# Informe Final — Sistema Inteligente de Detección de Transacciones Fraudulentas

**Asignatura:** Minería de Datos — Universidad Cooperativa de Colombia, Campus Pasto
**Docente:** PhD. Cristian Camilo Ordoñez Quintero
**Integrantes:** Cristian Gomes · Ricardo Arias · Mateo Guerrero · Kevin Agreda
**Fecha:** noviembre de 2026

---

## 1. Introducción y objetivo
El fraude con tarjetas de crédito causa pérdidas millonarias y es difícil de detectar porque es un evento **muy raro**. El objetivo del proyecto es desarrollar un sistema inteligente que, a partir de las características de una transacción, estime su **probabilidad de fraude** y genere **alertas** que un analista pueda revisar, usando técnicas de *machine learning* (Random Forest, XGBoost) y aprendizaje profundo (red neuronal MLP), con balanceo de clases mediante **SMOTE**.

## 2. Datos
- **Fuente:** Credit Card Fraud Detection — Kaggle (mlg-ulb/creditcardfraud). Transacciones de tarjetahabientes europeos durante dos días de septiembre de 2013.
- **Tamaño:** 284.807 transacciones y 31 columnas: `Time`, `V1`–`V28` (componentes PCA anonimizadas), `Amount` y `Class`.
- **Desbalance:** 492 fraudes (**0,173 %**).
- **Calidad:** sin valores nulos y con 1.081 filas duplicadas, que se eliminaron (quedan 283.726 filas y 473 fraudes).

## 3. Metodología (CRISP-DM)
1. **EDA:** distribución de clases, monto y hora por clase, correlación con la clase y densidades de las variables más discriminantes (gráficos 01–06).
2. **Preprocesamiento:** se reemplazó `Time` por `Hora` del día; `Amount` y `Hora` se escalaron con RobustScaler (por los valores atípicos); división estratificada en entrenamiento (64 %), validación (16 %) y prueba (20 %).
3. **Balanceo con SMOTE** dentro de un `Pipeline` de imbalanced-learn, aplicado **solo al entrenamiento** para evitar fuga de datos. Así se pasa de 302 a 181.282 fraudes de entrenamiento (gráfico 07).
4. **Modelos:** Regresión Logística (línea base), Random Forest, XGBoost y una red neuronal MLP (64-32-16). Random Forest y XGBoost se entrenaron también **sin** SMOTE para medir su efecto.
5. **Evaluación** en el conjunto de prueba con Precision, Recall, F1, ROC-AUC, PR-AUC y matriz de confusión (gráficos 08–10).
6. **Optimización del umbral** con predicciones *out-of-fold* (validación cruzada de 5 folds): se elige el umbral con máximo F1 sujeto a Recall ≥ 80 % (gráfico 11).
7. **Interpretabilidad** con la importancia de variables (gráfico 12) y **validación cruzada** de 5 folds para medir la estabilidad.
8. **Despliegue** en una aplicación web con Streamlit.

## 4. Resultados

### 4.1 Con umbral 0,5 (conjunto de prueba)
| Modelo | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| XGBoost (sin SMOTE) | 0,986 | 0,737 | 0,843 | 0,970 | 0,834 |
| Random Forest (sin SMOTE) | 0,958 | 0,726 | 0,826 | 0,956 | 0,799 |
| Random Forest + SMOTE | 0,815 | 0,789 | 0,802 | 0,974 | 0,809 |
| XGBoost + SMOTE | 0,735 | 0,789 | 0,761 | 0,967 | 0,808 |
| Red Neuronal MLP + SMOTE | 0,805 | 0,695 | 0,746 | 0,924 | 0,732 |
| Regresión Logística + SMOTE | 0,049 | 0,874 | 0,092 | 0,955 | 0,676 |

### 4.2 Con el umbral optimizado (conjunto de prueba)
| Modelo | Umbral | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| **XGBoost + SMOTE (final)** | 0,968 | 0,985 | 0,695 | 0,815 | 66 | 1 | 29 |
| Random Forest + SMOTE | 0,617 | 0,886 | 0,737 | 0,805 | 70 | 9 | 25 |
| Red Neuronal MLP + SMOTE | 0,214 | 0,744 | 0,705 | 0,724 | 67 | 23 | 28 |

### 4.3 Validación cruzada (5 folds)
| Modelo | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Random Forest + SMOTE | 0,820 ± 0,028 | 0,976 ± 0,012 | 0,836 ± 0,032 |
| XGBoost + SMOTE | 0,780 ± 0,025 | 0,975 ± 0,008 | 0,846 ± 0,031 |

## 5. Aplicación web (despliegue)
`streamlit run 05_Aplicacion_Web/app.py`
- **📊 Dashboard de fraude:** indicadores (alertas, fraudes detectados, falsas alarmas, dinero protegido), distribución de probabilidades, matriz de confusión interactiva y fraude por hora.
- **🚨 Alertas en tiempo real:** simula un flujo de transacciones que el modelo evalúa en vivo; muestra notificaciones, una gráfica de riesgo continua y un registro de alertas con nivel de riesgo.
- **🔎 Transacciones sospechosas:** tabla filtrable por probabilidad, monto y hora; nivel de riesgo; perfil de la transacción frente al perfil típico de fraude; descarga en CSV.
- **🧪 Analizar transacción:** evaluación de una transacción (editable) con los tres modelos, o de un lote desde un archivo CSV.
- **📈 Desempeño de los modelos:** métricas, curvas ROC y Precision–Recall, matriz de confusión, importancia de variables y validación cruzada.
- El **umbral de alerta** y el **modelo** se pueden cambiar desde la barra lateral.


## 6. Conclusiones

### Resultados del modelo final — **XGBoost + SMOTE** (umbral 0,968), conjunto de prueba (56.746 transacciones nunca vistas, 95 fraudes)

| Precision | Recall | F1 | ROC-AUC | PR-AUC | Fraudes detectados | Falsas alarmas |
|---|---|---|---|---|---|---|
| **0,985** | **0,695** | **0,815** | **0,967** | **0,808** | 66 de 95 | **1** de 56.651 legítimas |

1. **El problema es extremadamente desbalanceado** (0,17 % de fraude). La *accuracy* no sirve: un modelo que nunca detecta fraude tendría 99,83 %. Por eso usamos Precision, Recall, F1, ROC-AUC, PR-AUC y la matriz de confusión.
2. **SMOTE** se aplicó correctamente **solo sobre el entrenamiento** (dentro de un Pipeline), igualando 181.282 legítimas con 181.282 fraudes (302 reales + sintéticos). Con umbral 0,5, SMOTE **sube el Recall** (0,73 → 0,79 en Random Forest y 0,74 → 0,79 en XGBoost) pero también las falsas alarmas.
3. **Análisis crítico de SMOTE:** la PR-AUC (independiente del umbral) es casi igual con y sin SMOTE (≈ 0,80–0,83). SMOTE no agrega información nueva; mueve el punto de operación del modelo hacia "detectar más". Por eso **calibrar el umbral después del balanceo es indispensable**.
4. **Random Forest y XGBoost** superan con claridad a la regresión logística (que con umbral 0,5 lanza más de 1.600 falsas alarmas) y a la red neuronal MLP (ROC-AUC 0,924). En datos tabulares, los **ensambles de árboles** fueron los mejores.
5. **Validación cruzada (5 folds):** Random Forest + SMOTE → F1 0,820 ± 0,028 y ROC-AUC 0,976 ± 0,012; XGBoost + SMOTE → F1 0,780 ± 0,025 y ROC-AUC 0,975 ± 0,008. La variación entre folds es baja, así que **los modelos son estables**.
6. **Umbral:** se eligió con predicciones *out-of-fold* (sin tocar la prueba), maximizando F1 con la condición de Recall ≥ 80 %. En prueba el Recall fue 69,5 %, algo menor que en validación cruzada (≈ 80 %). Con solo 95 fraudes en prueba, cada fraude vale 1 punto porcentual de Recall, y el umbral 0,5 lograría 79 % de Recall a cambio de 27 falsas alarmas. **El umbral es una decisión de negocio** y por eso se puede ajustar en vivo desde la aplicación web.
7. **Impacto de negocio:** el sistema protege ≈ **€ 8.877 de € 14.766 (60 %)** del dinero en fraude del conjunto de prueba, con **1 sola falsa alarma**: casi cada alerta que revisa el analista es un fraude real.
8. **Interpretabilidad:** V14, V4, V10, V12 y V17 son las variables más importantes para ambos ensambles, igual que en el EDA. Esto confirma que los modelos aprendieron patrones reales.
9. **Patrones de negocio:** el fraude se concentra en la **madrugada (2–4 a. m.)**, con una tasa de hasta 1,7 % frente a 0,17 % en promedio, y en **montos bajos** (mediana 9,25 € frente a 22 €), típicos de "pruebas" con tarjetas robadas.

## 7. Trabajo futuro
- Ajustar hiperparámetros con búsqueda bayesiana (Optuna) y probar LightGBM/CatBoost.
- Comparar SMOTE con otras técnicas (ADASYN, Borderline-SMOTE, SMOTE-Tomek, `scale_pos_weight`).
- Explicar cada alerta con valores SHAP y usar un umbral sensible al costo (monto de la transacción).
- Desplegar el modelo como API y conectarlo a un flujo real de transacciones (Kafka) con reentrenamiento periódico.

## 8. Despliegue
La aplicación web (`05_Aplicacion_Web/app.py`) incluye: **Dashboard de fraude**, **Alertas en tiempo real**, **Transacciones sospechosas**, análisis de transacciones individuales o por lote (CSV) y una página de desempeño de los modelos.
