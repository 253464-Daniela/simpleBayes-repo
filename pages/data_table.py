import streamlit as st
import pandas as pd

def mostrar():
    
    try:
        tipos = st.session_state['tipos']
    except KeyError:
        st.info("No se ha detectado la carga de datos, por favor cargue un archivo.")
        return
    
    # Cargar datos
    df = st.session_state['df']
    st.subheader("Vista previa de los datos")
    st.dataframe(df)
    st.subheader("Tipos de columna detectados")
    # Crear un DataFrame con los tipos
    tipos_df = pd.DataFrame({
        'Tipo de Columna': ['Fecha', 'Numérica', 'Categoría', 'Binaria'],
        'Columnas': [tipos['fecha'], tipos['numerica'], tipos['categorica'], tipos['binaria']]
    })
    
    st.dataframe(tipos_df)