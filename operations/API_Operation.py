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
    """
    try:
        df = pd.read_pickle("rag_dataframe.pkl")
        # --- MUDANÇA IMPORTANTE: Carregando o novo arquivo de embeddings ---
        embeddings = np.load("gemini_embeddings_001.npy")
        logging.info("Base de conhecimento (RAG) carregada com sucesso do cache.")
        return df, embeddings
    except FileNotFoundError:
        logging.error("Arquivos da base de conhecimento ('rag_dataframe.pkl' ou 'gemini_embeddings_001.npy') não encontrados.")
        return None, None
    except Exception as e:
        logging.error(f"Falha crítica ao carregar a base de conhecimento pré-processada: {e}")
        return None, None

class GeminiRAG:
    def __init__(self, api_key: str):
        """
        Inicializa o sistema RAG, configurando a API do Gemini e carregando a base de dados.
        """
        self.model_generator = None
        self._ready = False

        with st.spinner("Carregando base de conhecimento..."):
            self.rag_df, self.rag_embeddings = load_preprocessed_rag_base()

        if self.rag_df is None or self.rag_embeddings is None:
            st.error("ERRO CRÍTICO: Arquivos da base de conhecimento ('rag_dataframe.pkl' ou 'gemini_embeddings_001.npy') não encontrados. A funcionalidade de IA será desativada.")
            self.rag_df = pd.DataFrame()
            self.rag_embeddings = np.array([])
            return
        else:
            st.toast("Base de conhecimento carregada com sucesso.", icon="🧠")

        try:
            if not api_key:
                st.error("A chave da API do Gemini não foi fornecida.")
                raise ValueError("A chave da API não pode ser vazia.")
                
            genai.configure(api_key=api_key)
            self.model_generator = genai.GenerativeModel('gemini-2.5-pro')
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
        Encontra os chunks mais relevantes usando Gemini/embedding-001.
        """
        if not self.is_ready():
            return "Base de conhecimento indisponível."

        try:
            result = genai.embed_content(
                model='models/embedding-001',
                content=query_text
            )
            query_embedding = np.array(result['embedding']).reshape(1, -1)
            # --- FIM DA MUDANÇA ---
            
            similarities = cosine_similarity(query_embedding, self.rag_embeddings)[0]
            top_k_indices = similarities.argsort()[-top_k:][::-1]
            relevant_chunks = self.rag_df.iloc[top_k_indices]
            
            context = "\n\n---\n\n".join(relevant_chunks['Answer_Chunk'].tolist())
            return context
        except Exception as e:
            st.error(f"Erro durante a busca semântica com o Gemini. Detalhes: {e}")
            return "Erro ao buscar informações relevantes."

    def answer_question(self, question: str) -> tuple[str, float]:
        """
        Orquestra o processo: busca e geração, ambos com Gemini.
        """
        if not self.is_ready():
            return "O sistema de IA não está operacional.", 0

        start_time = time.time()
        relevant_context = self._find_relevant_chunks(question, top_k=7)
        
        if "Erro" in relevant_context or not relevant_context.strip():
            answer = "Não foi possível consultar a base de conhecimento ou encontrar informações relevantes para responder à sua pergunta."
        else:
            prompt = f"""
            **Persona:** Você é um Consultor Especialista em Normas Regulamentadoras e Saúde e Segurança do Trabalho. Sua comunicação é didática, clara e completa.

            **Missão Crítica:** Sua missão é fornecer uma resposta completa, detalhada e bem estruturada à **Pergunta do Usuário**, baseando-se **única e exclusivamente** nas informações contidas no **Contexto Relevante**.

            **INSTRUÇÕES DETALHADAS PARA A RESPOSTA:**

            1.  **Síntese Abrangente:** Analise **TODOS** os trechos do contexto fornecido. Se múltiplos trechos abordam o mesmo tópico, sintetize as informações para construir uma resposta coesa e abrangente, conectando as ideias.

            2.  **Elaboração e Detalhamento:** Não se limite a uma resposta curta. Elabore sobre os pontos encontrados, explique os conceitos chave, detalhe os processos ou requisitos mencionados e, se o contexto permitir, cite exemplos ou condições específicas. O objetivo é educar o usuário sobre o tema.

            3.  **Estrutura e Clareza:** Organize sua resposta de forma lógica. Utilize parágrafos para separar ideias e, quando apropriado, use listas (bullet points) para apresentar itens, etapas ou requisitos de forma clara e fácil de ler. Use **negrito** para destacar os termos técnicos ou os pontos mais importantes.

            4.  **Fidelidade Absoluta (REGRA INQUEBRÁVEL):** Se o contexto fornecido não contém informações suficientes para responder à pergunta do usuário, sua única ação é responder com a seguinte frase: *"Com base estrita no contexto fornecido, não há informações detalhadas sobre o tópico solicitado."* **NÃO** invente informações ou use conhecimento externo.

            ---
            **Contexto Relevante (Sua única fonte da verdade):**
            {relevant_context}
            ---

            **Pergunta do Usuário:**
            {question}

            **Sua Resposta (Siga as instruções para uma resposta detalhada e estruturada):**
            """
            
            try:
                response = self.model_generator.generate_content(prompt)
                answer = response.text
            except Exception as e:
                st.error(f"Erro ao gerar a resposta com o modelo Gemini: {e}")
                answer = "Ocorreu um erro ao tentar gerar a resposta final."

        end_time = time.time()
        elapsed_time = end_time - start_time
        return answer, elapsed_time
