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
import clases.ollama_insights as ollama
import clases.bayesGenerator as bg


# Inicializar el generador de insights 
@st.cache_resource
def init_gemini():
    return GeminiInsightGenerator()

def init_ollama():
    return ollama.LlamaInsightGenerator()

def mostrar():
    try:
        tipos = st.session_state['tipos']
    except KeyError:
        st.info("No se ha detectado la carga de datos, por favor cargue un archivo.")
        return
    # Selección de variable objetivo
    #gemini = init_gemini()
    ollama = init_ollama()
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
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Calculo de probabilidades",
            "Visualizaciones", 
            "Gráfico temporal",
            "Clasificador Naive Bayes simple"
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
                            f'P(B)': p_evid,
                            f'P(B|A)': p_evid_dado_fallo,
                            f'P(A|B) Bayes': p_fallo_dado_evid_bayes
                        })
                
                if resultados:
                    print(resultados)
                    df_resultados = pd.DataFrame(resultados)
                    st.subheader("Resultados de probabilidades condicionales")
                    st.dataframe(df_resultados)
                    
                    # Gráfico de comparación
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df_resultados['Evidencia'], y=[p_fallo]*len(df_resultados), name='P(B)', marker_color='lightgray'))
                    #fig.add_trace(go.Bar(x=df_resultados['Evidencia'], y=df_resultados[f'P({target}|{col}) empírica'], name=f'P({target}|{col}) empírica', marker_color='skyblue'))
                    fig.add_trace(go.Bar(x=df_resultados['Evidencia'], y=df_resultados['P(A|B) Bayes'], name=f'P(A|B) Bayes', marker_color='orange'))
                    fig.update_layout(title='Comparación de probabilidades', xaxis_title='Evidencia', yaxis_title='Probabilidad', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button("Generar insights"): 
                        # --------------------------------------------------------
                        # Insights con  IA
                        evidencias = ollama.generar_lista_evidencias(
                            df=df,
                            target=target,
                            tipos=tipos,
                            p_fallo=p_fallo,
                            resultados_condicionales=resultados
                        )
                        # Insights con Gemini IA
                        # --------------------------------------------------------
                        st.subheader("Insights generados por IA")
                        # Reemplaza la instancia de GeminiInsightGenerator por LlamaInsightGenerator
                        insights = ollama.generar_insights(p_fallo, target, df, evidencias, tipos)
                        # Generar insights
                        
                        # Mostrar insights
                        for insight in insights:
                            st.write(insight)   
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
            #nbSimple.ejecutar_clasificador_manual(df, target, tipos)            
            bg.main()
    else:
        st.info("No hay suficientes datos para generar insights. Prueba con otro conjunto.")