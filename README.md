# AI Analyzer

Um analisador interativo que utiliza a IA do Google Gemini para responder perguntas com base em uma base de conhecimento interna (RAG).

## Começando

Estas instruções fornecerão uma cópia do projeto em funcionamento em sua máquina local.

### Pré-requisitos

- Python 3.8 ou superior
- Uma chave de API do Google AI Studio (Gemini).
- Arquivos de base de conhecimento pré-processados: `rag_dataframe.pkl` e `rag_embeddings.npy` no diretório raiz do projeto.

### Instalação

1.  Clone o repositório e navegue até o diretório do projeto.

2.  Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  Crie um arquivo `requirements.txt` com o conteúdo abaixo e instale as dependências:
    ```
    streamlit
    google-generativeai
    pandas
    numpy
    scikit-learn
    ```
    ```bash
    pip install -r requirements.txt
    ```

4.  Configure sua chave de API do Gemini. Crie uma pasta chamada `.streamlit` na raiz do seu projeto. Dentro dela, crie um arquivo chamado `secrets.toml` e adicione sua chave:
    ```toml
    # .streamlit/secrets.toml
    GEMINI_API_KEY="SUA_CHAVE_API_AQUI"
    ```

### Executando o Aplicativo

Para iniciar o aplicativo, execute o seguinte comando no seu terminal:

```bash
streamlit run main.py
