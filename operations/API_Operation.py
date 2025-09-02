import os
from dotenv import load_dotenv
import ollama
from pdfminer.high_level import extract_text as pdfminer_extract_text
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import time
import tempfile
import numpy as np
import streamlit as st
import re
import os

class PDFQA:
    def __init__(self):
        self.model = 'gpt-oss:20b'
        self.embedding_model = 'nomic-embed-text:latest'

    # ------------------Função para extrair texto de um PDF-----------------------
    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                text = pdfminer_extract_text(file)
            return self.clean_text(text)  # Alterado para self.clean_text
        except Exception as e:
            st.error(f"Erro ao extrair texto do PDF {pdf_path}: {str(e)}")
            return ""

    #-----------------Função para limpar o texto-------------------------
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s,.!?\'\"-]', '', text)
        return text.strip()

    # -----------------Função para dividir o texto em chunks menores----------------
    def split_text(self, text, chunk_size=5000):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_text(text)

    # ------------Função para gerar embeddings----------------------------------
    def generate_embeddings(self, text_chunks):
        """Gera embeddings para os chunks de texto."""
        embeddings = []
        try:
            for chunk in text_chunks:
                embedding = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=chunk
                )
                embeddings.append(embedding['embedding'])
            return np.array(embeddings)
        except Exception as e:
            st.error(f"Erro ao gerar embeddings: {str(e)}")
            return None

    # -------------Função para encontrar o trecho mais similar------------------------
    def find_most_similar(self, query_embedding, document_embeddings):
        try:
            similarities = cosine_similarity(query_embedding.reshape(1, -1), document_embeddings)[0]
            most_similar_idx = np.argmax(similarities)
            return most_similar_idx, similarities[most_similar_idx]
        except Exception as e:
            st.error(f"Erro ao encontrar trecho mais similar: {str(e)}")
            return 0, 0

    #----------------- Função para fazer perguntas ao modelo ----------------------
    def ask_llm(self, context, question):
        try:
            st.info("Enviando pergunta para o modelo...")
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': f'''
                        Contexto: {context}

                        Pergunta: {question}

                        Por favor, forneça uma resposta detalhada, precisa e abrangente, baseando-se estritamente nas informações do contexto fornecido. 
                        Elabore sua resposta com exemplos, explicações e detalhes relevantes. 
                        Se houver informações contraditórias entre os chunks, mencione isso na sua resposta.
                        Se for necessário calcular alguma data de vencimento, data de início, data de término, etc, calcule com base na pergunta.
                        '''
                    },
                    {
                        'role': 'user',
                        'content': question
                    }
                ]
            )
            st.success("Resposta recebida do modelo.")
            return response['message']['content']
        except Exception as e:
            st.error(f"Erro ao obter resposta do modelo: {str(e)}")
            return None

    # -------------------Função para obter o contexto relevante-----------------
    def get_relevant_context(self, chunks, query, k=20):
        query_embedding = self.generate_embeddings([query])
        if query_embedding is None:
            return ""
        
        chunk_embeddings = self.generate_embeddings([chunk.page_content for chunk in chunks])
        if chunk_embeddings is None:
            return ""

        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        most_similar_indices = similarities.argsort()[-k:][::-1]
        
        relevant_chunks = [chunks[i].page_content for i in most_similar_indices]
        
        sorted_chunks = [f"[Chunk {i+1}]\n{chunk}\n---" for i, chunk in enumerate(relevant_chunks)]
        context = "\n".join(sorted_chunks)
        
        return context

    # -------------------Função principal para processar PDFs e responder perguntas---------------
    def answer_question(self, pdf_files, question):
        start_time = time.time()

        try:
            with st.spinner("Carregando e processando PDFs..."):
                chunks = self.load_and_process_pdfs(pdf_files)
            st.success(f"PDFs processados. Número de chunks: {len(chunks)}")
            
            with st.spinner("Obtendo contexto relevante..."):
                relevant_context = self.get_relevant_context(chunks, question)
                st.info("Contexto relevante obtido com sucesso.")
            
            with st.spinner("Gerando resposta com o modelo..."):
                answer = self.ask_llm(relevant_context, question)
                st.info("Resposta gerada com sucesso.")
            st.success("Resposta gerada com sucesso.")
            
            end_time = time.time()
            elapsed_time = end_time - start_time

            return answer, elapsed_time
        except Exception as e:
            st.error(f"Erro inesperado ao processar a pergunta: {str(e)}")
            st.exception(e)
            return f"Ocorreu um erro ao processar a pergunta: {str(e)}", 0
    

    #-----------Função para carregar dados do PDF temporariamente no cache---------------
    @st.cache_data
    def load_and_process_pdfs(_self, pdf_files):
        """Carrega e processa os PDFs."""
        documents = []
        for pdf_file in pdf_files:
            pdf_content = pdf_file.read()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_content)
                temp_file_path = temp_file.name
            
            try:
                loader = PyPDFLoader(temp_file_path)
                documents.extend(loader.load())
            finally:
                os.unlink(temp_file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks



