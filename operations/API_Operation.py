import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re

logging.basicConfig(level=logging.INFO)

@st.cache_data(ttl=3600)
def load_preprocessed_rag_base() -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """
    Carrega o DataFrame e os embeddings pré-processados de arquivos locais.
    """
    try:
        df = pd.read_pickle("rag_dataframe.pkl")
        embeddings = np.load("gemini_embeddings_001.npy")
        logging.info("Base de conhecimento (RAG) carregada com sucesso do cache.")
        return df, embeddings
    except FileNotFoundError:
        logging.error("Arquivos da base de conhecimento não encontrados.")
        return None, None
    except Exception as e:
        logging.error(f"Falha crítica ao carregar a base de conhecimento: {e}")
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
            st.error("ERRO CRÍTICO: Arquivos da base de conhecimento não encontrados.")
            self.rag_df = pd.DataFrame()
            self.rag_embeddings = np.array([])
            return
        else:
            # --- MUDANÇA 1: LISTA DE COLUNAS CORRIGIDA ---
            # Usando os nomes de coluna corretos do seu CSV.
            required_cols = ['Norma_Referencia', 'Section_Number', 'Answer_Chunk']
            if not all(col in self.rag_df.columns for col in required_cols):
                st.error(f"ERRO CRÍTICO: O DataFrame não contém as colunas necessárias: {required_cols}. Verifique seu arquivo 'rag_dataframe.pkl'.")
                self._ready = False
                return
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
            st.error(f"Erro ao inicializar o modelo Gemini: {e}")
            self._ready = False
            raise

    def is_ready(self) -> bool:
        """Verifica se o sistema RAG está pronto para uso."""
        return self._ready

    def _find_relevant_chunks(self, query_text: str, top_k: int = 5) -> str:
        """
        Encontra os chunks mais relevantes e retorna o texto junto com sua referência normativa.
        """
        if not self.is_ready():
            return "Base de conhecimento indisponível."

        try:
            result = genai.embed_content(model='models/embedding-001', content=query_text)
            query_embedding = np.array(result['embedding']).reshape(1, -1)
            
            similarities = cosine_similarity(query_embedding, self.rag_embeddings)[0]
            top_k_indices = similarities.argsort()[-top_k:][::-1]
            
            relevant_chunks_df = self.rag_df.iloc[top_k_indices]
            
            formatted_chunks = []
            for index, row in relevant_chunks_df.iterrows():
                # --- MUDANÇA 2: FORMATAÇÃO DA REFERÊNCIA CORRIGIDA ---
                # Usando os nomes de coluna corretos para criar a string da fonte.
                reference = f"{row['Norma_Referencia']} - {row['Section_Number']}"
                chunk_text = row['Answer_Chunk']
                formatted_chunks.append(f"[Fonte: {reference}]\n{chunk_text}")

            context = "\n\n---\n\n".join(formatted_chunks)
            return context
        except Exception as e:
            st.error(f"Erro durante a busca semântica com o Gemini. Detalhes: {e}")
            return "Erro ao buscar informações relevantes."

    def _is_reference_query(self, question: str) -> bool:
        """Verifica se a pergunta do usuário é um pedido de referências."""
        question_lower = question.lower()
        keywords = ['referência', 'referencias', 'norma', 'normas', 'fonte', 'fontes', 'cláusula', 'clausula', 'item', 'itens', 'seção', 'section']
        return any(keyword in question_lower for keyword in keywords)

    def answer_question(self, question: str) -> tuple[str, float]:
        """
        Orquestra a resposta. Primeiro, detecta a intenção do usuário.
        """
        if not self.is_ready():
            return "O sistema de IA não está operacional.", 0

        start_time = time.time()
        
        relevant_context = self._find_relevant_chunks(question, top_k=10)
        
        if "Erro" in relevant_context or not relevant_context.strip():
            answer = "Não foi possível consultar a base de conhecimento ou encontrar informações relevantes."
        
        elif self._is_reference_query(question):
            st.info("Intenção detectada: Pedido de referências normativas.")
            references = re.findall(r'\[Fonte: (.*?)\]', relevant_context)
            
            if references:
                unique_references = sorted(list(set(references)), key=references.index)
                answer = "### Referências Normativas Encontradas:\n\n"
                answer += "\n".join([f"- {ref}" for ref in unique_references])
            else:
                answer = "Nenhuma referência normativa específica foi encontrada para os termos da sua busca."
        
        else:
            prompt = f"""
            **Persona:** Você é um Consultor Especialista em Normas Regulamentadoras, cuja maior prioridade é a precisão e a rastreabilidade da informação. Sua comunicação é didática e **sempre referenciada**.
            **Missão Crítica:** Fornecer uma resposta completa e detalhada à **Pergunta do Usuário**, baseando-se **única e exclusivamente** nas informações do **Contexto Relevante**. O aspecto mais importante da sua tarefa é citar as fontes de cada informação.
            **Formato do Contexto:** Cada trecho de informação no contexto é precedido por sua fonte no formato `[Fonte: Norma - Referência]`.
            **INSTRUÇÕES DETALHADAS PARA A RESPOSTA:**
            1. **Síntese e Elaboração:** Analise todos os trechos do contexto. Sintetize as informações para construir uma resposta coesa e detalhada, explicando os conceitos chave.
            2. **Citação Obrigatória:** Ao formular sua resposta, você **DEVE** citar a(s) fonte(s) normativa(s) de onde extraiu a informação. Integre a citação de forma natural no texto.
            3. **Estrutura Clara:** Organize a resposta de forma lógica, usando parágrafos, listas e **negrito** para destacar termos importantes.
            4. **Seção de Fontes:** Ao final de **TODA** a sua resposta, adicione uma seção chamada "**Fontes Consultadas**" e liste todas as referências `[Fonte: ...]` que você utilizou para construir a resposta.
            5. **Fidelidade Absoluta (REGRA INQUEBRÁVEL):** Se o contexto não contém informações para responder à pergunta, responda apenas: *"Com base estrita no contexto fornecido, não há informações sobre o tópico solicitado."*
            ---
            **Contexto Relevante (Sua única fonte da verdade):**
            {relevant_context}
            ---
            **Pergunta do Usuário:**
            {question}
            **Sua Resposta (Siga TODAS as instruções, incluindo a citação no texto e a lista de fontes ao final):**
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
