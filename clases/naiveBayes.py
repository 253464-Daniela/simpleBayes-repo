import numpy as np
import pandas as pd
import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

class NaiveBayesManual:
    """
    Implementación manual de Naive Bayes para clasificación binaria
    Maneja variables numéricas (Gaussianas) y categóricas
    """
    
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.numeric_stats = {}
        self.categorical_probs = {}
        self.numeric_cols = []
        self.categorical_cols = []
        
    def fit(self, X_num, X_cat, y):
        # Guardar nombres de columnas
        self.numeric_cols = X_num.columns.tolist() if X_num is not None else []
        self.categorical_cols = X_cat.columns.tolist() if X_cat is not None else []
        
        # Clases únicas
        self.classes = np.unique(y)
        
        # Calcular probabilidades previas P(Clase)
        total = len(y)
        for clase in self.classes:
            self.priors[clase] = np.sum(y == clase) / total
        
        # Calcular estadísticas para variables numéricas
        if X_num is not None:
            self.numeric_stats = {}
            for clase in self.classes:
                X_clase = X_num[y == clase]
                self.numeric_stats[clase] = {}
                for col in X_num.columns:
                    mean = X_clase[col].mean()
                    std = X_clase[col].std()
                    std = max(std, 1e-6)
                    self.numeric_stats[clase][col] = (mean, std)
        
        # Calcular probabilidades para variables categóricas
        if X_cat is not None:
            self.categorical_probs = {}
            for clase in self.classes:
                X_clase = X_cat[y == clase]
                self.categorical_probs[clase] = {}
                for col in X_cat.columns:
                    valores_unicos = X_cat[col].unique()
                    conteos = X_clase[col].value_counts()
                    
                    probs = {}
                    for valor in valores_unicos:
                        conteo = conteos.get(valor, 0)
                        probs[valor] = (conteo + 1) / (len(X_clase) + len(valores_unicos))
                    
                    self.categorical_probs[clase][col] = probs
    
    def _gaussian_prob(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
    
    def predict_proba(self, X_num, X_cat):
        n_samples = len(X_num) if X_num is not None else len(X_cat)
        probas = np.zeros((n_samples, len(self.classes)))
        
        for i in range(n_samples):
            for idx, clase in enumerate(self.classes):
                prob = np.log(self.priors[clase])
                
                if X_num is not None:
                    for col in self.numeric_cols:
                        x_val = X_num.iloc[i][col]
                        if pd.notna(x_val):
                            mean, std = self.numeric_stats[clase][col]
                            p = self._gaussian_prob(x_val, mean, std)
                            prob += np.log(p + 1e-10)
                
                if X_cat is not None:
                    for col in self.categorical_cols:
                        x_val = X_cat.iloc[i][col]
                        if pd.notna(x_val):
                            if x_val in self.categorical_probs[clase][col]:
                                p = self.categorical_probs[clase][col][x_val]
                            else:
                                p = 1e-6
                            prob += np.log(p)
                
                probas[i, idx] = prob
        
        probas = np.exp(probas)
        probas = probas / probas.sum(axis=1, keepdims=True)
        return probas
    
    def predict(self, X_num, X_cat):
        probas = self.predict_proba(X_num, X_cat)
        return np.argmax(probas, axis=1), probas[:, 1]

def ejecutar_clasificador_manual(df, target, tipos):
    """
    Función principal para ejecutar el clasificador manual en Streamlit
    ENTRENAMIENTO AUTOMÁTICO - sin botón
    """
    st.subheader("Clasificador Naive Bayes Manual")
    
    # ------------------------------------------------------------------
    # 1. Limpiar datos: asegurar que columnas numéricas sean float
    # ------------------------------------------------------------------
    df_clean = df.copy()
    cols_numericas_original = [col for col in tipos['numerica'] if col != target]
    for col in cols_numericas_original:
        if df_clean[col].dtype == 'bool':
            df_clean[col] = df_clean[col].astype(float)
        elif df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            except:
                pass
        # Si sigue sin ser numérico, forzar a float (puede generar NaN)
        if df_clean[col].dtype not in ['float64', 'int64']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # ------------------------------------------------------------------
    # 2. Preparar variables predictoras y objetivo
    # ------------------------------------------------------------------
    cols_numericas = [col for col in tipos['numerica'] if col != target]
    cols_categoricas = [col for col in tipos['categorica'] if col != target]
    cols_binarias = [col for col in tipos['binaria'] if col != target]
    
    
    X_num = df_clean[cols_numericas].copy() if cols_numericas else None
    
    if cols_binarias:
        cols_categoricas.extend(cols_binarias)
    
    X_cat = df_clean[cols_categoricas].copy() if cols_categoricas else None
    y = df_clean[target].copy()
    
    # ------------------------------------------------------------------
    # 3. Mostrar información básica
    # ------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Variables numéricas", len(cols_numericas) if cols_numericas else 0)
    with col2:
        st.metric("Variables categóricas", len(cols_categoricas) if cols_categoricas else 0)
    with col3:
        st.metric("Total muestras", len(y))
    
    # ------------------------------------------------------------------
    # 4. ID único para la sesión (persistente)
    # ------------------------------------------------------------------
    if 'nb_session_id' not in st.session_state:
        st.session_state.nb_session_id = str(np.random.randint(0, 1000000))
    session_id = st.session_state.nb_session_id
    
    # ------------------------------------------------------------------
    # 5. Slider para test_size
    # ------------------------------------------------------------------
    test_size = st.slider(
        "Tamaño del conjunto de prueba",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        key=f"test_size_{session_id}"
    )
    
    # ------------------------------------------------------------------
    # 6. División train/test
    # ------------------------------------------------------------------
    if X_num is not None and X_cat is not None:
        X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
            X_num, X_cat, y, test_size=test_size, random_state=42, stratify=y
        )
    elif X_num is not None:
        X_num_train, X_num_test, y_train, y_test = train_test_split(
            X_num, y, test_size=test_size, random_state=42, stratify=y
        )
        X_cat_train = X_cat_test = None
    else:
        X_cat_train, X_cat_test, y_train, y_test = train_test_split(
            X_cat, y, test_size=test_size, random_state=42, stratify=y
        )
        X_num_train = X_num_test = None
    
    # ------------------------------------------------------------------
    # 7. Mostrar distribución de clases (opcional en expander)
    # ------------------------------------------------------------------
    with st.expander("Ver distribución de clases"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Entrenamiento:**")
            train_dist = y_train.value_counts().sort_index()
            st.dataframe(pd.DataFrame({
                'Clase': train_dist.index,
                'Conteo': train_dist.values,
                'Porcentaje': (train_dist.values / len(y_train) * 100).round(1)
            }))
        with col2:
            st.write("**Prueba:**")
            test_dist = y_test.value_counts().sort_index()
            st.dataframe(pd.DataFrame({
                'Clase': test_dist.index,
                'Conteo': test_dist.values,
                'Porcentaje': (test_dist.values / len(y_test) * 100).round(1)
            }))
    
    # ------------------------------------------------------------------
    # 8. Entrenamiento automático del modelo
    # ------------------------------------------------------------------
    entrenamiento_msg = st.empty()
    entrenamiento_msg.info("Entrenando modelo automáticamente...")
    
    nb_manual = NaiveBayesManual()
    nb_manual.fit(X_num_train, X_cat_train, y_train)
    y_pred, y_proba = nb_manual.predict(X_num_test, X_cat_test)
    
    acc = accuracy_score(y_test, y_pred)
    sensibilidad = recall_score(y_test, y_pred, pos_label=1)
    especificidad = recall_score(y_test, y_pred, pos_label=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Guardar en session_state
    st.session_state['modelo_manual'] = nb_manual
    st.session_state['metricas_manual'] = {
        'accuracy': acc,
        'sensibilidad': sensibilidad,
        'especificidad': especificidad,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'cm': cm
    }
    
    entrenamiento_msg.success("Modelo entrenado automáticamente!")
    
    # ------------------------------------------------------------------
    # 9. Mostrar métricas y gráficos
    # ------------------------------------------------------------------
    st.write("### Métricas de rendimiento")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{acc:.3f}")
    with col2:
        st.metric("Sensibilidad", f"{sensibilidad:.3f}")
    with col3:
        st.metric("Especificidad", f"{especificidad:.3f}")
    with col4:
        st.metric("Muestras prueba", len(y_test))
    
    st.write("### Matriz de Confusión")
    fig_cm = ff.create_annotated_heatmap(
        cm,
        x=['Predicho 0', 'Predicho 1'],
        y=['Real 0', 'Real 1'],
        colorscale='Blues',
        showscale=True,
        annotation_text=cm.astype(str)
    )
    fig_cm.update_layout(width=500, height=500)
    st.plotly_chart(fig_cm, use_container_width=True)
    
    vn, fp, fn, vp = cm.ravel()
    st.markdown(f"""
    - ✅ **Verdaderos Negativos:** {vn}
    - ❌ **Falsos Positivos:** {fp}
    - ❌ **Falsos Negativos:** {fn}
    - ✅ **Verdaderos Positivos:** {vp}
    """)
    
    st.write("### Distribución de Probabilidades")
    fig_hist = px.histogram(
        x=y_proba,
        nbins=30,
        title="P(Fallo=1 | Datos)",
        labels={'x': 'Probabilidad de fallo'}
    )
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    with st.expander("Ver comparación de predicciones (primeros 20)"):
        comparacion = pd.DataFrame({
            'Real': y_test.values[:20],
            'Predicho': y_pred[:20],
            'Probabilidad': y_proba[:20].round(3),
            'Acierto': y_test.values[:20] == y_pred[:20]
        })
        def colorear_filas(row):
            if row['Acierto']:
                return ['background-color: #d4edda'] * len(row)
            return ['background-color: #f8d7da'] * len(row)
        styled_df = comparacion.style.apply(colorear_filas, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    
    # ------------------------------------------------------------------
    # 10. Sección de predicción con nuevos datos
    # ------------------------------------------------------------------
    st.write("---")
    st.write("### Probar modelo con nuevos datos")
    
    cols = st.columns(2)
    nuevos_datos_num = {}
    nuevos_datos_cat = {}
    
    with cols[0]:
        st.write("**Variables numéricas:**")
        for i, col in enumerate(cols_numericas):
            min_val = float(df_clean[col].min())
            max_val = float(df_clean[col].max())
            mean_val = float(df_clean[col].mean())
            
            persist_key = f"num_persist_{col}"  # Clave para almacenar el valor persistente
            
            # Inicializar el valor persistente si no existe, asegurando que sea float
            if persist_key not in st.session_state:
                st.session_state[persist_key] = mean_val
            else:
                # Si existe, aseguramos que sea float (por si acaso)
                try:
                    val = float(st.session_state[persist_key])
                except (TypeError, ValueError):
                    val = mean_val
                st.session_state[persist_key] = val
            
            current_val = st.session_state[persist_key]
            
            nuevos_datos_num[col] = st.number_input(
                label=f"{col}",
                min_value=min_val,
                max_value=max_val,
                value=current_val,
                step=0.1,
                format="%.2f",
                key=f"num_{col}_{session_id}",
                on_change=lambda col=col, pk=persist_key: st.session_state.__setitem__(
                    pk, 
                    float(st.session_state[f"num_{col}_{session_id}"])
                )
            )
    with cols[1]:
        st.write("**Variables categóricas:**")
        for i, col in enumerate(cols_categoricas):
            persist_key = f"cat_persist_{col}"
            
            valores = df_clean[col].unique()
            
            if f"cat_{col}" not in st.session_state:
                st.session_state[f"cat_{col}"] = valores[0]
            
            nuevos_datos_cat[col] = st.selectbox(
                f"{col}", 
                valores,
                index=list(valores).index(st.session_state[f"cat_{col}"]) if st.session_state[f"cat_{col}"] in valores else 0,
                key=f"cat_{col}_{session_id}",
                on_change=lambda col=col, pk=persist_key: st.session_state.__setitem__(
                    pk, 
                    st.session_state[f"cat_{col}_{session_id}"]
                )
            )
    
    if st.button("Predecir", key=f"predict_btn_{session_id}"):
        modelo = st.session_state['modelo_manual']
        nuevo_num = pd.DataFrame([nuevos_datos_num]) if nuevos_datos_num else None
        nuevo_cat = pd.DataFrame([nuevos_datos_cat]) if nuevos_datos_cat else None
        
        pred, proba = modelo.predict(nuevo_num, nuevo_cat)
        
        col1, col2 = st.columns(2)
        with col1:
            if pred[0] == 1:
                st.error("**FALLO PREDICHO**")
            else:
                st.success("**NO FALLO PREDICHO**")
        with col2:
            st.metric("Probabilidad de fallo", f"{proba[0]:.3f}")