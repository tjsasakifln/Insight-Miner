# Insight Miner

## Problema

Empresas coletam feedback de clientes de diversas fontes, mas extrair insights acionáveis desses dados pode ser um processo lento e manual. A análise manual de milhares de reviews é impraticável, e as equipes de produto e marketing perdem a oportunidade de entender rapidamente as necessidades e frustrações dos clientes.

## Solução

O Insight Miner é um dashboard interativo que automatiza a análise de feedback de clientes. Ele utiliza processamento de linguagem natural (NLP) para identificar tendências de sentimento, extrair os principais tópicos de discussão e visualizar os dados de forma clara e concisa. Com o Insight Miner, as empresas podem:

*   **Centralizar o feedback:** Faça o upload de um arquivo CSV com os reviews dos seus clientes.
*   **Analisar o sentimento:** Entenda rapidamente se os clientes estão satisfeitos ou insatisfeitos.
*   **Extrair tópicos:** Descubra os principais temas de conversa dos seus clientes.
*   **Visualizar os dados:** Explore os dados através de gráficos interativos e nuvens de palavras.
*   **Obter resumos com IA:** Gere resumos concisos dos insights com o poder do GPT-4.

## Como Executar Localmente

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/seu-usuario/insight-miner.git
    cd insight-miner
    ```

2.  **Crie e ative o ambiente virtual:**

    ```bash
    python -m venv .venv
    # No Windows
    .venv\Scripts\activate
    # No macOS/Linux
    source .venv/bin/activate
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Crie um arquivo `.env`** na raiz do projeto e adicione sua chave da API da OpenAI:

    ```
    OPENAI_API_KEY=sua-chave-aqui
    ```

5.  **Execute o aplicativo:**

    ```bash
    streamlit run scripts/app.py
    ```

## Integrações Futuras

*   **Google NLP:** Integração com a API de Natural Language do Google para análise de sentimento e extração de entidades mais avançada.
*   **Amazon Comprehend:** Utilização do Amazon Comprehend para análise de texto em larga escala.

---

**Quer isso rodando na sua empresa? Fale comigo.**
