import streamlit as st
from operations.API_Operation import PDFQA

def page_1():

    st.title("Prompts and Responses")

    # Criar uma instância da classe PDFQA
    pdf_qa_instance = PDFQA()

    # Movendo o upload de arquivos e a mensagem de sucesso para a barra lateral
    with st.sidebar:
        uploaded_files = st.file_uploader("Escolha os arquivos PDF", accept_multiple_files=True, type="pdf")
        if uploaded_files:
            st.success(f"{len(uploaded_files)} arquivo(s) carregado(s) com sucesso!")

    # Conteúdo principal
    if uploaded_files:
        question = st.text_input("Digite sua pergunta sobre os PDFs:")

        if st.button("Enviar pergunta"):
            if uploaded_files and question:
                with st.spinner("Processando..."):
                    answer, elapsed_time = pdf_qa_instance.answer_question(uploaded_files, question)  # Usar a instância
                    st.subheader("Resposta:")
                    st.write(answer)
                    st.info(f"Tempo de resposta: {elapsed_time:.2f} segundos")
            else:
                st.warning("Por favor, digite uma pergunta.")
    else:
        st.info("Por favor, faça o upload de pelo menos um arquivo PDF na barra lateral.")


def show_about_page():
    st.title("Sobre o Analisador de PDFs")
    st.write("""
    Este aplicativo permite carregar múltiplos arquivos PDF e fazer perguntas sobre seu conteúdo.
    Utilizamos tecnologia de processamento de linguagem natural para analisar os documentos e fornecer respostas relevantes.
    
    Desenvolvido por [CRISTIAN CARLOS]
    Versão: 1.0
    """)

