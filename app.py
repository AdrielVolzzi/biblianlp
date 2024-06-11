import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import unidecode
import string
import openai

# Baixar as stopwords em português
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# Função para preprocessar o texto
def preprocess_text(text):
    text = text.lower()
    text = unidecode.unidecode(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Carregar a base de dados de versículos
@st.cache_data
def load_data():
    bible_df = pd.read_csv('bible_verses.csv')
    return bible_df

bible_df = load_data()
bible_df['processed_text'] = bible_df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(bible_df['processed_text'])

def search_verses(user_input):
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-20:][::-1]
    results = bible_df.iloc[similar_indices]
    results = results.drop_duplicates(subset=['book', 'chapter', 'verse']).head(5)
    return results

def get_other_versions(book, chapter, verse):
    return bible_df[(bible_df['book'] == book) & (bible_df['chapter'] == chapter) & (bible_df['verse'] == verse)]

def summarize_chapter(book, chapter):
    chapter_texts = bible_df[(bible_df['book'] == book) & (bible_df['chapter'] == chapter)]['text'].tolist()
    chapter_text = ' '.join(chapter_texts)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você é um assistente útil que resume capítulos da Bíblia."},
            {"role": "user", "content": f"Resuma o capítulo seguinte: {chapter_text}"}
        ],
        max_tokens=400
    )
    return response.choices[0].message['content'].strip()

def explain_verse(book, chapter, verse):
    verse_text = bible_df[(bible_df['book'] == book) & (bible_df['chapter'] == chapter) & (bible_df['verse'] == verse)]['text'].values[0]
    chapter_texts = bible_df[(bible_df['book'] == book) & (bible_df['chapter'] == chapter)]['text'].tolist()
    chapter_text = ' '.join(chapter_texts)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você é um assistente útil que explica versículos da Bíblia no contexto do capítulo."},
            {"role": "user", "content": f"Explique o seguinte versículo no contexto do capítulo: '{verse_text}' dentro do capítulo: {chapter_text}"}
        ],
        max_tokens=400
    )
    return response.choices[0].message['content'].strip()

def main():
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Ir para", ["Busca de Versículos", "Análise de Versículos"])

    if page == "Busca de Versículos":
        st.title('Busca de Versículos Bíblicos')
        user_input = st.text_area('Insira um texto para buscar versículos semelhantes:')
        search_button = st.button('Pronto')
        
        if search_button and user_input:
            results = search_verses(user_input)
            st.write("Versículos mais próximos ao texto inserido:")
            for _, row in results.iterrows():
                st.write(f"{row['book']} {row['chapter']}:{row['verse']} ({row['version']}): {row['text']}")
            st.session_state.results = results

    elif page == "Análise de Versículos":
        st.title('Análise de Versículos Bíblicos')
        
        book = st.selectbox('Selecione o livro:', sorted(bible_df['book'].unique()))
        chapter = st.number_input('Selecione o capítulo:', min_value=1, step=1)
        verse = st.number_input('Selecione o versículo:', min_value=1, step=1)
        analyze_button = st.button('Analisar')

        if analyze_button:
            st.subheader("Versões")
            other_versions = get_other_versions(book, chapter, verse)
            for _, row in other_versions.iterrows():
                st.write(f"{row['version']}: {row['text']}")
            
            st.subheader("Contexto do Capítulo")
            chapter_summary = summarize_chapter(book, chapter)
            st.write(chapter_summary)
            
            st.subheader("Explicação")
            explanation = explain_verse(book, chapter, verse)
            st.write(explanation)

if __name__ == "__main__":
    openai.api_key = ''  # chave de API da OpenAI
    main()
