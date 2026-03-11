import streamlit as st


st.set_page_config(page_title="Analizador Bayesiano de Anomalías", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)
def navbar_superior():
    """Crea una barra de navegación superior usando tabs"""
    
    # CSS personalizado para mejorar apariencia
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 4px 4px 0px 0px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Crear tabs como navegación
    tab1, tab2, tab3, tab4 = st.tabs([
        "Inicio", 
        "Carga de archivo", 
        "Visualizacion de datos", 
        "Resultados"
    ])
    
    with tab1:
        import presentation
        presentation.mostrar()
        
    with tab2:
        import pages.carga_archivos as carga_archivos
        carga_archivos.mostrar()
        
    with tab3:
        import pages.data_table as data_table
        data_table.mostrar()
        
    with tab4:
        import pages.results as results
        results.mostrar()

# Usar la navbar
navbar_superior()