import streamlit as st
from operations.API_Operation import GeminiRAG

def page_1():
    st.title("Base de Conhecimento Interativa com IA")
    st.markdown("Faça uma pergunta e a IA buscará as informações mais relevantes em nossa base de conhecimento para construir uma resposta.")

    # Inicializa a instância do GeminiRAG. 
    # A classe lida com o carregamento dos dados e a exibição de mensagens de erro/sucesso.
    try:
        if 'rag_instance' not in st.session_state:
            st.session_state.rag_instance = GeminiRAG()
        
        rag_instance = st.session_state.rag_instance

        # Verifica se a base de conhecimento foi carregada com sucesso
        if not rag_instance.is_ready():
            st.error("A funcionalidade de perguntas e respostas está desativada devido a um erro no carregamento da base de conhecimento ou da chave de API. Verifique os logs e o arquivo de secrets.")
            return # Interrompe a execução da página se a base não estiver pronta

    except Exception as e:
        st.error(f"Ocorreu um erro crítico ao inicializar o sistema de IA: {e}")
        st.info("Verifique se a chave GEMINI_API_KEY está configurada corretamente no arquivo .streamlit/secrets.toml")
        return

    # Conteúdo principal
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
    Versão: 2.0 (Gemini RAG)
    """)
