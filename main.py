import streamlit as st
from operations.frontend import page_1, show_about_page

st.set_page_config(page_title="AI Analyzer", page_icon="⚡", layout="wide")

def main():
    try:
        # Default role to admin to show all pages
        st.session_state.role = 'admin'
        
        pages = ['Análise com IA', 'Sobre']
        if st.session_state.role == 'admin':
            pages.append('Cadastro de Usuário')

        page = st.sidebar.radio("Selecione a página", pages)

        if page == "Análise com IA":
                page_1()
        elif page == "Sobre":
                show_about_page()
        

    except Exception as e:
            st.error(f"Ocorreu um erro inesperado: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
