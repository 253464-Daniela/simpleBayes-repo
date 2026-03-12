import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def calcular_naive_bayes(df, clase_col, clase_val, evidencias_dict):
    total_n = len(df)
    prior = len(df[df[clase_col] == clase_val]) / total_n
    if prior == 0: return 0
    
    likelihood_acumulada = 1.0
    df_clase = df[df[clase_col] == clase_val]
    
    for var, val_observado in evidencias_dict.items():
        coincidencias = len(df_clase[df_clase[var] == val_observado])
        p_evidencia_dado_clase = (coincidencias + 1) / (len(df_clase) + df[var].nunique())
        likelihood_acumulada *= p_evidencia_dado_clase
        
    return prior * likelihood_acumulada

def main():
    st.subheader("Clasificador Naive Bayes")
    df_original = st.session_state["df"]
    vars_dict = st.session_state["tipos"]
    
    #df = discretizar_numericas(df_original, vars_dict["numerica"])
    df = df_original
    
    clase_col = st.selectbox("Variable objetivo", options=vars_dict["binaria"])
    binarias = df[clase_col].unique()
    objetivo_positivo = st.selectbox("Objetivo positivo", options=binarias)
    objetivo_negativo = [opt for opt in binarias if opt != objetivo_positivo][0]
    variables_detectadas = vars_dict["categorica"] + vars_dict["numerica"] + vars_dict["binaria"]
    variables_detectadas = [v for v in variables_detectadas if v != clase_col] 
    evidencias_seleccionadas = st.multiselect("Seleccionar evidencias", options=variables_detectadas)
    
    if len(evidencias_seleccionadas) > 0:
        st.subheader("Valores posibles")
    input_evidencias = {}
    for var in evidencias_seleccionadas:
        if var in vars_dict["numerica"]:
            input_evidencias[var] = st.select_slider(f"Valor de {var}", options=sorted(df[var].unique()))
        else:
            input_evidencias[var] = st.selectbox(f"Valor de {var}", options=df[var].unique())

    if clase_col and evidencias_seleccionadas:
        prob_pos = calcular_naive_bayes(df, clase_col, objetivo_positivo, input_evidencias)
        prob_neg = calcular_naive_bayes(df, clase_col, objetivo_negativo, input_evidencias)
        
        suma = prob_pos + prob_neg
        p_final_pos = (prob_pos / suma) if suma > 0 else 0
        p_final_neg = (prob_neg / suma) if suma > 0 else 0


        with st.container(border=False):
            st.subheader("Diagnóstico de Probabilidad")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Datos: dos valores (positivo y negativo)
                valores = [p_final_pos, p_final_neg]
                etiquetas = [f"Prob. {objetivo_positivo}", f"Prob. {objetivo_negativo}"]
                colores = ["#5985dc", '#eb6960']
                
                fig_prob =  go.Figure(data=[go.Pie(
                    labels=etiquetas,
                    values=valores,
                    hole=0.6,  # tamaño del hueco central (0 = pastel, 1 = agujero total)
                    marker=dict(colors=colores),
                    textinfo='percent',  # muestra porcentajes dentro del gráfico
                    insidetextorientation='radial',
                    showlegend=False
                )])
                # Añadir anotación en el centro con la probabilidad positiva
                fig_prob.add_annotation(
                    text=f"{p_final_pos:.2%}",
                    x=0.5, y=0.5,
                    font_size=30,
                    showarrow=False
                )
                
                fig_prob.update_layout(xaxis_range=[0, 1], title="Resultado de Inferencia")
                st.plotly_chart(fig_prob, use_container_width=True)

            with col2:
                st.metric(f"Probabilidad de {objetivo_positivo}", f"{p_final_pos:.2%}")
                st.write(f"Basado en {len(evidencias_seleccionadas)} evidencias.")

        # B. EVALUACIÓN DEL MODELO (Matriz y Métricas)
        with st.container(border=False):
            st.subheader("Evaluación del Clasificador (Rendimiento)")
            
            y_real = df[clase_col]
            y_pred = []
            
            for idx, row in df.iterrows():
                evidencias_fila = {v: row[v] for v in evidencias_seleccionadas}
                p_pos = calcular_naive_bayes(df, clase_col, objetivo_positivo, evidencias_fila)
                p_neg = calcular_naive_bayes(df, clase_col, objetivo_negativo, evidencias_fila)
                y_pred.append(objetivo_positivo if p_pos >= p_neg else objetivo_negativo)
            
            # Cálculo de Matriz de Confusión
            tp = sum((y_real == objetivo_positivo) & (np.array(y_pred) == objetivo_positivo))
            tn = sum((y_real == objetivo_negativo) & (np.array(y_pred) == objetivo_negativo))
            fp = sum((y_real == objetivo_negativo) & (np.array(y_pred) == objetivo_positivo))
            fn = sum((y_real == objetivo_positivo) & (np.array(y_pred) == objetivo_negativo))

            # Métricas
            accuracy = (tp + tn) / len(df)
            sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
            especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{accuracy:.2%}")
            m2.metric("Sensibilidad", f"{sensibilidad:.2%}")
            m3.metric("Especificidad", f"{especificidad:.2%}")

            z = [[tn, fp], [fn, tp]]
            x_labels = [f"Pred: {objetivo_negativo}", f"Pred: {objetivo_positivo}"]
            y_labels = [f"Real: {objetivo_negativo}", f"Real: {objetivo_positivo}"]
            
            fig_conf = px.imshow(z, x=x_labels, y=y_labels, text_auto=True, color_continuous_scale='RdPu', title="Matriz de Confusión")
            st.plotly_chart(fig_conf, use_container_width=True)  

    else:
        st.warning("Selecciona la variable objetivo y al menos una evidencia.")