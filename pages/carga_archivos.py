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




def mostrar():
    # Configuración de la página
    st.markdown("Carga un archivo CSV y la aplicación detectará automáticamente tipos de columnas, calculará probabilidades condicionales y aplicará el teorema de Bayes.")

    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV, Excel o XLS",  # Texto principal
        key="file_uploader",
        type=["csv", "xlsx", "xls"],
        help="Formatos aceptados: CSV, Excel (XLSX, XLS)",  # Texto de ayuda
        accept_multiple_files=False
    )
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file
        es_valido, mensaje, df = LoadFiles.validar_archivo(uploaded_file)
        if es_valido:
            # Detectar tipos iniciales
            tipos = LoadFiles.detectar_tipos(df)
            
            st.session_state['tipos'] = tipos
            st.session_state['df'] = df
            st.success("Archivo cargado correctamente, dirigete a la página de resultados o a la de visualización para continuar.")
        else:
            # Mostrar mensaje de error
            st.error(f"❌ {mensaje}")
            
            # Opcional: Mostrar sugerencias
            with st.expander("Ver sugerencias"):
                st.write("""
                - Asegúrate de que el archivo tenga extensión .csv, .xlsx o .xls
                - Verifica que el archivo no esté corrupto
                - Comprueba que el archivo tenga datos en formato tabular
                - Si es CSV, verifica que esté correctamente delimitado
                """)
    else:
        st.info("Esperando la carga de un archivo CSV...")