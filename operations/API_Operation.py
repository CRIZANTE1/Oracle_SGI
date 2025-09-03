import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
import logging
import ollama 

logging.basicConfig(level=logging.INFO)

@st.cache_data(ttl=3600)
def load_preprocessed_rag_base() -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """
    Carrega o DataFrame e os embeddings pré-processados de arquivos locais.
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
        Inicializa o sistema RAG.
        - Carrega a base de dados local.
        - Configura a API do Gemini para a GERAÇÃO de respostas.
        """
        self.model_generator = None # Modelo para gerar a resposta final
        self._ready = False

        # Carrega a base de conhecimento primeiro
        with st.spinner("Carregando base de conhecimento..."):
            self.rag_df, self.rag_embeddings = load_preprocessed_rag_base()

        if self.rag_df is None or self.rag_embeddings is None:
            st.error("ERRO CRÍTICO: Arquivos da base de conhecimento ('rag_dataframe.pkl' ou 'rag_embeddings.npy') não encontrados. A funcionalidade de IA será desativada.")
            self.rag_df = pd.DataFrame()
            self.rag_embeddings = np.array([])
            return
        else:
            st.toast("Base de conhecimento carregada com sucesso.", icon="🧠")

        # Configura o modelo de GERAÇÃO (Google Gemini)
        try:
            if not api_key:
                st.error("A chave da API do Gemini não foi fornecida.")
                raise ValueError("A chave da API não pode ser vazia.")
                
            genai.configure(api_key=api_key)
            self.model_generator = genai.GenerativeModel('gemini-1.5-pro-latest')
            logging.info("Modelo Gerador (Gemini) configurado com sucesso.")
            self._ready = True

        except Exception as e:
            st.error(f"Erro ao inicializar o modelo Gemini. Verifique se a chave da API é válida. Detalhes: {e}")
            logging.error(f"Erro durante a inicialização do modelo Gemini: {e}")
            self._ready = False
            raise

    def is_ready(self) -> bool:
        """Verifica se o sistema RAG está pronto para uso."""
        return self._ready

    def _find_relevant_chunks(self, query_text: str, top_k: int = 5) -> str:
        """
        Encontra os chunks mais relevantes usando Ollama/nomic-embed-text.
        """
        if not self.is_ready():
            return "Base de conhecimento indisponível."

        try:

            response = ollama.embeddings(
                model='nomic-embed-text:latest',
                prompt=query_text
            )
            query_embedding = np.array(response['embedding']).reshape(1, -1)
            # --- FIM DA MUDANÇA ---
            
            similarities = cosine_similarity(query_embedding, self.rag_embeddings)[0]
            top_k_indices = similarities.argsort()[-top_k:][::-1]
            relevant_chunks = self.rag_df.iloc[top_k_indices]
            
            context = "\n\n---\n\n".join(relevant_chunks['Answer_Chunk'].tolist())
            return context
        except Exception as e:
            st.error(f"Erro ao conectar com o Ollama para busca. Verifique se o serviço Ollama está rodando. Detalhes: {e}")
            return "Erro ao buscar informações relevantes."

    def answer_question(self, question: str) -> tuple[str, float]:
        """
        Orquestra o processo: busca com Ollama, geração com Gemini.
        """
        if not self.is_ready():
            return "O sistema de IA não está operacional.", 0

        start_time = time.time()
        
        relevant_context = self._find_relevant_chunks(question, top_k=7)
        
        if "Erro" in relevant_context or not relevant_context.strip():
            answer = "Não foi possível consultar a base de conhecimento ou encontrar informações relevantes para responder à sua pergunta."
        else:
            prompt = f"""
            **Persona:** Você é um Oráculo Analítico, especialista na norma ISO 45001.
            **Missão Crítica:** Sua tarefa é responder à **Pergunta do Usuário** usando **única e exclusivamente** as informações contidas no **Contexto Relevante**.
            **REGRAS DE OURO:**
            1. **SE A RESPOSTA ESTIVER NO CONTEXTO:** Responda à pergunta de forma clara, baseando-se estritamente nos trechos fornecidos.
            2. **SE A RESPOSTA NÃO ESTIVER NO CONTEXTO:** Responda com uma declaração clara de que a informação não foi encontrada. **NÃO** utilize conhecimento externo ou resuma conteúdo irrelevante. Exemplo: *"Com base estrita no contexto fornecido, não há informações sobre o tópico solicitado."*
            ---
            **Contexto Relevante (Sua única fonte da verdade):**
            {relevant_context}
            ---
            **Pergunta do Usuário:**
            {question}
            **Sua Resposta (Siga as Regras de Ouro):**
            """
            
            try:
                # Usando o modelo gerador do Gemini para a resposta final
                response = self.model_generator.generate_content(prompt)
                answer = response.text
            except Exception as e:
                st.error(f"Erro ao gerar a resposta com o modelo Gemini: {e}")
                answer = "Ocorreu um erro ao tentar gerar a resposta final."

        end_time = time.time()
        elapsed_time = end_time - start_time
        return answer, elapsed_time
