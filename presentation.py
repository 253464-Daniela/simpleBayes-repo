import streamlit as st
import clases.generate_data as generate_data


def mostrar():
    st.title("Analizador Bayesiano de Eventos Anómalos")
    # Configuración de la página
    #if st.button("Generar Datos"):
        #generate_data.generar_datos() 

    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR52-E4-0k-ayZm_xQPFM5bYRpgm12pegtrlA&s", width=200)
    with col2: 
        st.subheader("Realizado por:")
        st.caption("Daniela Michell Zúñiga Monterrosa")
        st.caption("Julián López Zambrano")
        st.caption("Maria Elizabeth Velazquez Escobar")
        st.caption("Ingenieria en tegnologías de la información e innovación digital")
