import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)

@st.cache_data(ttl=3600)
def load_preprocessed_rag_base() -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """
    Carrega o DataFrame e os embeddings pré-processados de arquivos locais.
    Retorna None em caso de falha para que a classe possa lidar com o erro.
    """
    try:
        df = pd.read_pickle("rag_dataframe.pkl")
        embeddings = np.load("rag_embeddings.npy")
        logging.info("Base de conhecimento (RAG) carregada com sucesso do cache.")
        return df, embeddings
    except FileNotFoundError:
        logging.error("Arquivos da base de conhecimento ('rag_dataframe.pkl' ou 'rag_embeddings.npy') não encontrados.")
        return None, None
    except Exception as e:
        logging.error(f"Falha crítica ao carregar a base de conhecimento pré-processada: {e}")
        return None, None

class GeminiRAG:
    def __init__(self, api_key: str):
        """
        Inicializa o sistema RAG, configurando a API do Gemini e carregando a base de dados.
        """
        self.model = None
        self._ready = False # Inicia como Falso por padrão

        if not api_key:
            st.error("A chave da API fornecida está vazia.")
            raise ValueError("A chave da API não pode ser vazia.")
        
        # Carrega a base de conhecimento PRIMEIRO, antes de configurar a API
        # Isso economiza chamadas de API se os arquivos locais estiverem faltando.
        with st.spinner("Carregando base de conhecimento..."):
            self.rag_df, self.rag_embeddings = load_preprocessed_rag_base()

        # Verifica o resultado do carregamento e mostra as mensagens apropriadas (SUA LÓGICA IMPLEMENTADA AQUI)
        if self.rag_df is None or self.rag_embeddings is None:
            st.error("ERRO CRÍTICO: Arquivos da base de conhecimento ('rag_dataframe.pkl' ou 'rag_embeddings.npy') não encontrados. A funcionalidade de IA será desativada.")
            # Garante que os atributos sejam DataFrames vazios para evitar erros posteriores
            self.rag_df = pd.DataFrame()
            self.rag_embeddings = np.array([])
            # self._ready continua False, o que desativa a UI principal
            return # Interrompe a inicialização aqui
        else:
            st.toast("Base de conhecimento carregada com sucesso.", icon="🧠")

        # Se a base de conhecimento carregou, prossiga com a configuração da API
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
            logging.info("Modelo Gemini configurado com sucesso.")
            # Somente se a base e a API estiverem OK, o sistema está pronto
            self._ready = True

        except Exception as e:
            st.error(f"Erro ao inicializar o modelo Gemini. Verifique se a chave da API é válida. Detalhes: {e}")
            logging.error(f"Erro durante a inicialização da classe GeminiRAG: {e}")
            # Se a API falhar, o sistema também não está pronto
            self._ready = False
            raise

    def is_ready(self) -> bool:
        """Verifica se o sistema RAG está pronto para uso."""
        return self._ready

    def _find_relevant_chunks(self, query_text: str, top_k: int = 5) -> str:
        """
        Encontra os chunks mais relevantes na base de conhecimento para uma dada pergunta.
        """
        if not self.is_ready():
            return "Base de conhecimento indisponível."

        try:
            query_embedding_result = genai.embed_content(
                model='models/text-embedding-004',
                content=[query_text],
                task_type="RETRIEVAL_QUERY"
            )
            query_embedding = np.array(query_embedding_result['embedding']).reshape(1, -1)
            
            similarities = cosine_similarity(query_embedding, self.rag_embeddings)[0]
            top_k_indices = similarities.argsort()[-top_k:][::-1]
            relevant_chunks = self.rag_df.iloc[top_k_indices]
            
            context = "\n\n---\n\n".join(relevant_chunks['Answer_Chunk'].tolist())
            return context
        except Exception as e:
            st.warning(f"Erro durante a busca na base de conhecimento: {e}")
            return "Erro ao buscar informações relevantes."

    def answer_question(self, question: str) -> tuple[str, float]:
        """
        Orquestra o processo de responder a uma pergunta usando RAG.
        """
        if not self.is_ready():
            return "O sistema de IA não está operacional.", 0

        start_time = time.time()
        
        relevant_context = self._find_relevant_chunks(question, top_k=7)
        
        if "indisponível" in relevant_context or "Erro" in relevant_context:
            answer = "Não foi possível consultar a base de conhecimento para responder à sua pergunta."
        else:
            prompt = f"""
            Você é um assistente especialista. Sua tarefa é responder à pergunta do usuário de forma precisa e detalhada, baseando-se ESTREITAMENTE no contexto fornecido abaixo. Não utilize conhecimento externo.

            **Contexto da Base de Conhecimento:**
            ---
            {relevant_context}
            ---

            **Pergunta do Usuário:**
            {question}

            **Sua Resposta:**
            """
            
            try:
                response = self.model.generate_content(prompt)
                answer = response.text
            except Exception as e:
                st.error(f"Erro ao gerar a resposta com o modelo de IA: {e}")
                answer = "Ocorreu um erro ao tentar gerar a resposta final."

        end_time = time.time()
        elapsed_time = end_time - start_time

        return answer, elapsed_time

