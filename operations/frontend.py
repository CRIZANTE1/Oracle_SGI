import streamlit as st
from operations.API_Operation import GeminiRAG

def page_1():
    st.title("Base de Conhecimento Interativa com IA")

    # --- Seção da Barra Lateral para a Chave da API ---
    with st.sidebar:
        st.header("Configuração")
        # Usamos type="password" para mascarar a chave
        api_key_input = st.text_input(
            "Insira sua Chave da API Gemini", 
            type="password",
            help="Sua chave não será armazenada. Ela é usada apenas durante esta sessão."
        )

        # Botão para salvar a chave na sessão
        if st.button("Salvar Chave"):
            if api_key_input:
                st.session_state.gemini_api_key = api_key_input
                # Limpa a instância antiga caso a chave seja alterada
                if 'rag_instance' in st.session_state:
                    del st.session_state.rag_instance
                st.success("Chave da API salva para esta sessão!")
                st.rerun() # Força o recarregamento da página para usar a nova chave
            else:
                st.warning("Por favor, insira uma chave da API.")

    # --- Conteúdo Principal da Página ---

    # Verifica se a chave API foi inserida e salva na sessão
    if 'gemini_api_key' not in st.session_state or not st.session_state.gemini_api_key:
        st.info("👋 Bem-vindo! Por favor, insira sua chave da API do Google Gemini na barra lateral para começar a interagir com a IA.")
        st.stop() # Interrompe a execução do resto da página

    # Tenta inicializar a instância do RAG apenas uma vez por sessão (com a chave fornecida)
    try:
        if 'rag_instance' not in st.session_state:
            with st.spinner("Inicializando o sistema de IA..."):
                st.session_state.rag_instance = GeminiRAG(api_key=st.session_state.gemini_api_key)
        
        rag_instance = st.session_state.rag_instance

        if not rag_instance.is_ready():
            st.error("A funcionalidade de IA está desativada. Verifique se os arquivos da base de conhecimento existem.")
            st.stop()

    except Exception as e:
        st.error(f"Falha na inicialização do sistema de IA. Verifique se sua chave da API é válida.")
        # Limpa a chave inválida para que o usuário possa tentar novamente
        del st.session_state.gemini_api_key
        st.stop()
    
    # Se tudo deu certo, mostra a interface de perguntas e respostas
    st.markdown("Faça uma pergunta e a IA buscará as informações mais relevantes em nossa base de conhecimento para construir uma resposta.")
    
    question = st.text_input("Digite sua pergunta sobre a base de conhecimento:", key="question_input")

    if st.button("Enviar Pergunta", type="primary"):
        if question:
            with st.spinner("Buscando informações e gerando resposta..."):
                answer, elapsed_time = rag_instance.answer_question(question)
                st.subheader("Resposta da IA:")
                st.markdown(answer)
                st.success(f"Tempo de resposta: {elapsed_time:.2f} segundos")
        else:
            st.warning("Por favor, digite uma pergunta.")


def show_about_page():
    st.title("Sobre o Analisador com IA")
    st.write("""
    Este aplicativo utiliza o poder do Google Gemini para responder perguntas complexas com base em uma base de conhecimento interna.
    A tecnologia de Retrieval-Augmented Generation (RAG) permite que a IA encontre os trechos de informação mais relevantes para fundamentar suas respostas.
    
    Desenvolvido por [CRISTIAN CARLOS]
    Versão: 2.1 (Gemini RAG com API Key na UI)
    """)
