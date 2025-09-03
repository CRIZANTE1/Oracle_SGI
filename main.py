import streamlit as st
from operations.frontend import page_1, show_about_page

# Configuração da página atualizada para refletir o tema
st.set_page_config(
    page_title="Oráculo de SSO", 
    page_icon=" ⚡", 
    layout="wide"
)

def main():
    try:
        # Lista de páginas simplificada, sem gerenciamento de usuários
        pages = ['Análise com IA', 'Sobre']

        # Navegação na barra lateral
        page = st.sidebar.radio("Selecione a página", pages)

        # Roteamento de páginas
        if page == "Análise com IA":
            page_1()
        elif page == "Sobre":
            show_about_page()

    except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
