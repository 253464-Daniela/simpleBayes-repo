import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, ConfusionMatrixDisplay
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime
from clases.simpleBayes import SimpleNaiveBayes
from clases.loadFiles import LoadFiles
from clases.gemini_insights import GeminiInsightGenerator
import clases.naiveBayes as nbSimple


# Inicializar el generador de insights 
@st.cache_resource
def init_gemini():
    return GeminiInsightGenerator()

def generar_lista_evidencias(df, target, tipos, p_fallo, num_evidencias, cat_evidencias, resultados_condicionales=None, modelo_metrics=None):
    """
    Genera una lista de evidencias (hallazgos) a partir de los datos y análisis.
    """
    evidencias = []
    
    # 1. Probabilidad a priori
    evidencias.append(f"La probabilidad a priori de {target} es {p_fallo:.3f}.")
    
    # 2. Estadísticas descriptivas de variables numéricas respecto a target
    num_cols = [col for col in tipos['numerica'] if col != target]
    for col in num_cols:
        media_global = df[col].mean()
        media_clase1 = df[df[target]==1][col].mean()
        media_clase0 = df[df[target]==0][col].mean()
        evidencias.append(f"Media de {col}: global={media_global:.2f}, cuando {target}=1: {media_clase1:.2f}, cuando {target}=0: {media_clase0:.2f}.")
    
    # 3. Distribución de variables categóricas
    cat_cols = [col for col in tipos['categorica'] if col != target] + tipos['binaria']
    for col in cat_cols:
        # Proporción de target=1 por cada categoría
        prop = df.groupby(col)[target].mean().sort_values(ascending=False)
        for cat, prob in prop.items():
            evidencias.append(f"Cuando {col} = {cat}, la probabilidad de {target}=1 es {prob:.3f}.")
    
    # 4. Resultados de probabilidades condicionales (si se proporcionan)
    if resultados_condicionales:
        for res in resultados_condicionales:
            evid = res['Evidencia']
            cond = res['Condición']
            p_bayes = list(res.values())[4]
            evidencias.append(f"Para la evidencia '{evid} {cond}'")
            if p_bayes:
                evidencias.append(f"Según Bayes, la probabilidad sería {p_bayes:.3f}.")
    
    # 5. Métricas del modelo (si se proporcionan)
    if modelo_metrics:
        acc = modelo_metrics.get('accuracy')
        sens = modelo_metrics.get('sensibilidad')
        espec = modelo_metrics.get('especificidad')
        if acc is not None:
            evidencias.append(f"El modelo Naive Bayes tiene una exactitud (accuracy) de {acc:.3f}.")
            evidencias.append(f"Sensibilidad (tasa de detección de {target}=1): {sens:.3f}.")
            evidencias.append(f"Especificidad (tasa de detección de {target}=0): {espec:.3f}.")
        # Matriz de confusión
        cm = modelo_metrics.get('cm')
        if cm is not None:
            vn, fp, fn, vp = cm.ravel()
            evidencias.append(f"Matriz de confusión: Verdaderos Negativos={vn}, Falsos Positivos={fp}, Falsos Negativos={fn}, Verdaderos Positivos={vp}.")
    
    return evidencias

def mostrar():
    try:
        tipos = st.session_state['tipos']
    except KeyError:
        st.info("No se ha detectado la carga de datos, por favor cargue un archivo.")
        return
    # Selección de variable objetivo
    gemini = init_gemini()
    df = st.session_state['df']
    todas_columnas = df.columns.tolist()
    filter_columnas = tipos['binaria']
    resultados = []
    
    if filter_columnas:
        target = st.selectbox("Selecciona la variable objetivo (evento anómalo)", filter_columnas)
    
        # --- Conversión robusta de la variable objetivo a binaria 0/1 ---
        with st.spinner("Procesando variable objetivo..."):
            try:
                # Si ya está en binarias, mapeamos por si acaso (ej. valores como 'sí' no numéricos)
                if target in tipos['binaria']:
                    df[target] = LoadFiles.mapear_binaria(df[target])
            except Exception as e:
                st.error(f"Error al procesar la variable objetivo: {e}")
                st.stop()
        
        # Recalcular tipos después de la conversión (para actualizar listas de evidencias)
        tipos = LoadFiles.detectar_tipos(df)
        
        # Probabilidad a priori P(Fallo)
        print(target)
        p_fallo = df[target].mean()
        st.metric(f"Probabilidad P({target})", f"{p_fallo:.4f}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Calculo de probabilidades",
            "Visualizaciones", 
            "Gráfico temporal",
            "Clasificador Naive Bayes simple",
            "Insights"
        ])
        
        with tab1:
            # --------------------------------------------------------
            # Selección de evidencias para cálculo condicional
            # --------------------------------------------------------
            st.subheader("Cálculo de probabilidades condicionales")
            evidencias_disponibles = [col for col in todas_columnas if col != target]
            
            # Clasificamos las evidencias según los tipos actualizados
            num_evidencias = [col for col in evidencias_disponibles if col in tipos['numerica']]
            cat_evidencias = [col for col in evidencias_disponibles if col in tipos['categorica']]
            bin_evidencias = [col for col in evidencias_disponibles if col in tipos['binaria']]
            
            st.write("Selecciona las variables de evidencia y configura los umbrales (para numéricas) o valores (para categóricas).")
            
            evidencias_seleccionadas = []
            umbrales = {}
            valores_cat = {}
            
            with st.expander("Variables numéricas"):
                for col in num_evidencias:
                    usar = st.checkbox(f"Incluir {col}", key=f"num_{col}")
                    if usar:
                        evidencias_seleccionadas.append(col)
                        default_thresh = df[col].median()
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        umbral = st.slider(f"Umbral para {col} (> ...)", min_val, max_val, float(default_thresh), key=f"thresh_{col}")
                        umbrales[col] = umbral
            
            with st.expander("Variables categóricas"):
                for col in cat_evidencias:
                    usar = st.checkbox(f"Incluir {col}", key=f"cat_{col}")
                    if usar:
                        evidencias_seleccionadas.append(col)
                        valores_unicos = df[col].dropna().unique()
                        valor = st.selectbox(f"Valor de {col} para condición", valores_unicos, key=f"val_{col}")
                        valores_cat[col] = valor
            
            with st.expander("Variables binarias (ya en 0/1)"):
                for col in bin_evidencias:
                    usar = st.checkbox(f"Incluir {col}", key=f"bin_{col}")
                    if usar:
                        evidencias_seleccionadas.append(col)
                        # Para binarias, consideramos el valor 1 como evidencia
                        umbrales[col] = 0.5  # umbral para >0.5 significa ==1
                        valores_cat[col] = 1
            
            # Si hay evidencias seleccionadas, calculamos las probabilidades
            if evidencias_seleccionadas:
                for col in evidencias_seleccionadas:
                    if col in num_evidencias or col in bin_evidencias:
                        p_fallo_dado_evid, p_evid_dado_fallo = SimpleNaiveBayes.prob_condicional(df, target, col, 'numerica', threshold=umbrales.get(col))
                        condicion = f">{umbrales[col]:.2f}" if col in num_evidencias else "=1"
                    else:  # categórica
                        p_fallo_dado_evid, p_evid_dado_fallo = SimpleNaiveBayes.prob_condicional(df, target, col, 'categorica', valor_cat=valores_cat.get(col))
                        condicion = f"={valores_cat[col]}"
                    
                    if p_fallo_dado_evid is not None:
                        # Calcular P(Evidencia)
                        if col in num_evidencias or col in bin_evidencias:
                            p_evid = (df[col] > umbrales[col]).mean()
                        else:
                            p_evid = (df[col] == valores_cat[col]).mean()
                        
                        # Aplicar Bayes
                        p_fallo_dado_evid_bayes = SimpleNaiveBayes.bayes_theorem(p_fallo, p_evid_dado_fallo, p_evid)
                        
                        resultados.append({
                            'Evidencia': col,
                            'Condición': condicion,
                            f'P({col})': p_evid,
                            #f'P({target}|{col}) empírica': p_fallo_dado_evid,
                            f'P({col}|{target})': p_evid_dado_fallo,
                            f'P({target}|{col}) Bayes': p_fallo_dado_evid_bayes
                        })
                
                if resultados:
                    df_resultados = pd.DataFrame(resultados)
                    st.subheader("Resultados de probabilidades condicionales")
                    st.dataframe(df_resultados)
                    
                    # Gráfico de comparación
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df_resultados['Evidencia'], y=[p_fallo]*len(df_resultados), name=f'P({col})', marker_color='lightgray'))
                    #fig.add_trace(go.Bar(x=df_resultados['Evidencia'], y=df_resultados[f'P({target}|{col}) empírica'], name=f'P({target}|{col}) empírica', marker_color='skyblue'))
                    fig.add_trace(go.Bar(x=df_resultados['Evidencia'], y=df_resultados[f'P({target}|{col}) Bayes'], name=f'P({target}|{col}) Bayes', marker_color='orange'))
                    fig.update_layout(title='Comparación de probabilidades', xaxis_title='Evidencia', yaxis_title='Probabilidad', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No se pudieron calcular probabilidades para las evidencias seleccionadas.")
        
        with tab2:
            # --------------------------------------------------------
            # Visualizaciones adicionales
            # --------------------------------------------------------
            st.subheader("Visualizaciones")
            
            # Histogramas de variables numéricas
            if tipos['numerica']:
                st.write("### Histogramas de variables numéricas")
                num_cols_plot = st.multiselect(
                    "Selecciona variables numéricas para histograma", 
                    tipos['numerica'], 
                    default=tipos['numerica'][:3]
                )
                
                for col in num_cols_plot:
                    fig = px.histogram(
                        df, 
                        x=col, 
                        nbins=30,
                        title=f'Histograma de {col}',
                        labels={col: col}
                    )
                    fig.update_layout(
                        xaxis_title=col,
                        yaxis_title="Frecuencia",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Gráfico temporal si hay columna de fecha
            if tipos['fecha']:
                st.write("### Gráfico temporal")
                fecha_col = st.selectbox("Selecciona columna de fecha", tipos['fecha'])
                num_col_temp = st.selectbox("Selecciona variable numérica para eje Y", tipos['numerica'] if tipos['numerica'] else ['Ninguna'])
                if num_col_temp != 'Ninguna':
                    df_temp = df.copy()
                    df_temp[fecha_col] = pd.to_datetime(df_temp[fecha_col])
                    fig = px.line(df_temp, x=fecha_col, y=num_col_temp, title=f'Evolución de {num_col_temp} en el tiempo')
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:            
            nbSimple.ejecutar_clasificador_manual(df, target, tipos)            
        
        with tab5:
            uploaded_file = st.session_state['uploaded_file']
            
            evidencias = generar_lista_evidencias(
                df=df,
                target=target,
                tipos=tipos,
                p_fallo=p_fallo,
                num_evidencias=num_evidencias,
                cat_evidencias=cat_evidencias,
                resultados_condicionales=resultados,  # de la pestaña de cálculo
                modelo_metrics=st.session_state.get('metricas_manual', None)
            )
            # Insights con Gemini IA
            # --------------------------------------------------------
            st.subheader("🤖 Insights generados por Gemini IA")
            
            # Generar insights
            insights = gemini.generar_insights(
                p_fallo=p_fallo,
                target=target,
                df=df,
                evidencias_impacto=evidencias,
                tipos=tipos,
                contexto_adicional={
                    'nombre_archivo': uploaded_file.name,
                    'tamaño_dataset': len(df)
                }
            )
            
            # Mostrar insights
            for insight in insights:
                st.write(insight)
            
            # Opción para regenerar
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🔄 Regenerar insights"):
                    st.rerun()
    else:
        st.info("No hay suficientes datos para generar insights. Prueba con otro conjunto.")