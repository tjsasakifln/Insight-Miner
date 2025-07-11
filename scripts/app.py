import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import openai
from dotenv import load_dotenv
import os

load_dotenv()

# Load the OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    st.set_page_config(layout="wide")
    st.title("Dashboard da Voz do Cliente")

    # File uploader
    uploaded_file = st.file_uploader(
        "Faça o upload de um arquivo CSV", type=["csv"],
        help="O arquivo CSV deve ter as colunas: reviewer_name, date, rating, review_text, source"
    )

    if uploaded_file is not None:
        # Load the data
        df = pd.read_csv(uploaded_file)

        # --- Data Processing ---
        # Sentiment Analysis
        sid = SentimentIntensityAnalyzer()
        df['sentiment'] = df['review_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
        df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

        # Topic Extraction
        @st.cache_data
        def get_topics(data, n_topics=5):
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            doc_term_matrix = vectorizer.fit_transform(data)
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(doc_term_matrix)
            return lda, vectorizer

        lda_model, vectorizer = get_topics(df['review_text'])

        def display_topics(model, feature_names, n_top_words):
            topics = []
            for topic_idx, topic in enumerate(model.components_):
                topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            return topics

        df['topic'] = lda_model.transform(vectorizer.transform(df['review_text'])).argmax(axis=1)
        topic_names = display_topics(lda_model, vectorizer.get_feature_names_out(), 5)
        df['topic_name'] = df['topic'].apply(lambda x: topic_names[x])


        # --- KPIs ---
        st.header("KPIs")
        total_reviews = len(df)
        average_rating = df['rating'].mean()
        positive_reviews_count = len(df[df['sentiment_label'] == 'positive'])
        negative_reviews_count = len(df[df['sentiment_label'] == 'negative'])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Reviews", total_reviews)
        col2.metric("Nota Média", f"{average_rating:.2f}")
        col3.metric("Reviews Positivos", positive_reviews_count)
        col4.metric("Reviews Negativos", negative_reviews_count)

        # --- Charts ---
        st.header("Análise Visual")

        # Sentiment Timeline
        df['date'] = pd.to_datetime(df['date'])
        sentiment_over_time = df.groupby('date')['sentiment'].mean().reset_index()
        st.subheader("Linha do Tempo do Sentimento")
        st.line_chart(sentiment_over_time.rename(columns={'date':'index'}).set_index('index'))

        # Topic Distribution
        topic_dist = df['topic_name'].value_counts()
        st.subheader("Distribuição de Tópicos")
        st.bar_chart(topic_dist)

        # Word Clouds
        st.subheader("Nuvens de Palavras")
        positive_reviews = " ".join(df[df['sentiment_label'] == 'positive']['review_text'])
        negative_reviews = " ".join(df[df['sentiment_label'] == 'negative']['review_text'])

        col1, col2 = st.columns(2)
        with col1:
            st.text("Tópicos Positivos")
            if positive_reviews:
                wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
                st.image(wordcloud_pos.to_array())
        with col2:
            st.text("Tópicos Negativos")
            if negative_reviews:
                wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews)
                st.image(wordcloud_neg.to_array())

        # --- GPT-4 Summary ---
        st.header("Resumo com GPT-4")
        if st.button("Gerar Resumo"):
            with st.spinner("Gerando resumo..."):
                prompt = f"Resuma os seguintes insights de reviews de clientes em três frases curtas: {df.to_string()}"
                try:
                    response = openai.Completion.create(
                        engine="text-davinci-003", # Or a different model
                        prompt=prompt,
                        max_tokens=150
                    )
                    st.success(response.choices[0].text.strip())
                except Exception as e:
                    st.error(f"Erro ao contatar a API da OpenAI: {e}")

        # --- Data Table ---
        st.header("Dados")
        st.dataframe(df)

if __name__ == "__main__":
    main()
