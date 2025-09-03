import streamlit as st
from operations.API_Operation import GeminiRAG

def page_1():
    st.title("Oráculo de SSO - Baseado na ISO 45001") # Título da página principal atualizado

    # --- Seção da Barra Lateral para a Chave da API ---
    with st.sidebar:
        st.header("Configuração")
        api_key_input = st.text_input(
            "Insira sua Chave da API Gemini", 
            type="password",
            help="Sua chave não será armazenada. Ela é usada apenas durante esta sessão."
        )

        if st.button("Salvar Chave"):
            if api_key_input:
                st.session_state.gemini_api_key = api_key_input
                if 'rag_instance' in st.session_state:
                    del st.session_state.rag_instance
                st.success("Chave da API salva para esta sessão!")
                st.rerun()
            else:
                st.warning("Por favor, insira uma chave da API.")

    # --- Conteúdo Principal da Página ---
    if 'gemini_api_key' not in st.session_state or not st.session_state.gemini_api_key:
        st.info("👋 Bem-vindo! Por favor, insira sua chave da API do Google Gemini na barra lateral para consultar a base de conhecimento sobre a ISO 45001.")
        st.stop()

    try:
        if 'rag_instance' not in st.session_state:
            with st.spinner("Inicializando o sistema de IA..."):
                st.session_state.rag_instance = GeminiRAG(api_key=st.session_state.gemini_api_key)
        
        rag_instance = st.session_state.rag_instance

        if not rag_instance.is_ready():
            st.error("A funcionalidade de IA está desativada. Verifique se os arquivos da base de conhecimento (`rag_dataframe.pkl`, `rag_embeddings.npy`) existem no diretório do projeto.")
            st.stop()

    except Exception as e:
        st.error(f"Falha na inicialização do sistema de IA. Verifique se sua chave da API é válida.")
        if 'gemini_api_key' in st.session_state:
            del st.session_state.gemini_api_key
        st.stop()
    
    st.markdown("Faça uma pergunta sobre os requisitos, cláusulas ou conceitos da norma **ISO 45001:2018**.")
    
    question = st.text_input("Ex: 'Quais são os requisitos para a política de SSO?' ou 'Explique a cláusula 8.1.2'", key="question_input")

    if st.button("Enviar Pergunta", type="primary"):
        if question:
            with st.spinner("Consultando a norma e gerando resposta..."):
                answer, elapsed_time = rag_instance.answer_question(question)
                st.subheader("Resposta do Oráculo:")
                st.markdown(answer)
                st.success(f"Tempo de resposta: {elapsed_time:.2f} segundos")
        else:
            st.warning("Por favor, digite uma pergunta.")


# --- FUNÇÃO 'SOBRE' TOTALMENTE REFEITA ---
def show_about_page():
    st.title("Sobre o Oráculo de Saúde e Segurança Ocupacional")
    st.markdown("""
    Bem-vindo ao assistente de IA especializado em Saúde e Segurança Ocupacional (SSO). 
    Esta ferramenta foi projetada para ser um recurso rápido e inteligente para profissionais, auditores e estudantes da área.
    """)

    st.header("Nossa Base de Conhecimento: A Norma ISO 45001")
    st.info("""
    O conhecimento deste "Oráculo" é estritamente fundamentado na norma técnica:

    **ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. NBR ISO 45001:2018 – Sistemas de gestão de saúde e segurança ocupacional – Requisitos com orientações para uso. Rio de Janeiro: ABNT, 2018.**

    A ISO 45001 é o padrão internacional para sistemas de gestão de SSO, projetado para proteger funcionários e visitantes de acidentes e doenças relacionadas ao trabalho. Ela substituiu a antiga norma OHSAS 18001.
    """)

    st.header("Como Funciona a Tecnologia?")
    st.markdown("""
    Quando você faz uma pergunta, a IA não responde com base em conhecimento genérico da internet. Em vez disso, ela utiliza a tecnologia de **Geração Aumentada por Recuperação (RAG)**, que funciona em duas etapas principais:

    1.  **Recuperação:** O sistema primeiro busca e recupera os trechos e cláusulas mais relevantes da norma ISO 45001 que se relacionam com a sua pergunta.
    2.  **Geração:** Em seguida, o modelo de linguagem avançado do Google (Gemini) utiliza esses trechos recuperados como contexto para formular uma resposta coesa, precisa e fácil de entender.

    Isso garante que as respostas sejam fiéis ao conteúdo da norma, aumentando a confiabilidade da informação.
    """)
    
    st.header("Para Quem é Este Aplicativo?")
    st.markdown("""
    *   **Auditores** de Sistemas de Gestão de SSO.
    *   **Profissionais de SST** (Técnicos, Engenheiros, Analistas).
    *   **Gestores e líderes** responsáveis pela implementação e manutenção da ISO 45001.
    *   **Estudantes e pesquisadores** da área de segurança do trabalho.
    """)

    st.warning("""
    **⚠️ Aviso Importante**

    Esta é uma ferramenta de auxílio e consulta rápida. As respostas da IA devem ser usadas como um ponto de partida e **não substituem a leitura completa da norma ou a consulta a um profissional qualificado**. Sempre verifique as informações críticas diretamente na fonte oficial.
    """)

    st.write("---")
    st.write("Desenvolvido por **[CRISTIAN CARLOS]**")
    st.write("Versão: 2.2 (Base de Conhecimento ISO 45001)")
