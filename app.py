import openai
import os
import json
from dotenv import load_dotenv
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Load the data
with open('article_embeddings.json', 'r') as file:
    article_embeddings = json.load(file)

with open('law_data.json', 'r') as file:
    law_data = json.load(file)
load_dotenv()  # This line loads the variables from .env

api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=api_key)


def get_embeddings(text):
    res = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return res.data[0].embedding

def calculate_similarities(query_vector, article_embeddings):
    query_vector = np.array(query_vector).reshape(1, -1)
    similarities = {}

    for title, article_vector in article_embeddings.items():
        article_vector = np.array(article_vector).reshape(1, -1)
        similarity = cosine_similarity(query_vector, article_vector)[0][0]
        similarities[title] = similarity

    return similarities

def get_article_content(title, data):
    # Retrieve the list of paragraphs for the given title, default to an empty list if the title is not found
    paragraphs = data.get(title, [])

    # Combine the title and the paragraphs into a single text string
    return title + '\n' + ' '.join(paragraphs)

def main():
    st.title("ChatG-TG")
    st.markdown("""
        ### Willkommen bei ChatG-TG!

Diese Anwendung ermöglicht es Ihnen, rechtliche Anfragen im Zusammenhang mit dem Thurgauer Lehrpersonalrecht zu stellen und auf Basis relevanter Gesetzestexte AI-generierte Antworten zu erhalten. Geben Sie Ihre Anfrage in das untenstehende Eingabefeld ein und klicken Sie auf 'Verarbeiten', um Ihre Antwort zu erhalten.

**So funktioniert's:**  
1. Geben Sie Ihre rechtliche Frage in das Eingabefeld ein.  
2. Klicken Sie auf den Button 'Verarbeiten', um die Anfrage zu starten.  
3. Die App analysiert Ihre Frage und findet relevante Gesetzesartikel.  
4. Sie erhalten eine AI-generierte Antwort basierend auf den gefundenen Artikeln.

    """)
    user_input = st.text_input("Ihre Frage zum Thurgauer Lehrpersonalrecht")
    if st.button("Bearbeiten") and user_input:
        query_vector = get_embeddings(user_input)
        similarities = calculate_similarities(query_vector, article_embeddings)
        sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        top_articles = sorted_articles[:5]
        combined_articles_text = "\n".join([get_article_content(title, law_data) for title, _ in top_articles])

        prompt = f"Frage: {user_input}\nNachfolgend findest Du fünf Gesetzesartikel. Prüfe ob die Artikel relevant sind und beantworte die Frage basierend auf den relevanten Artikel. Wenn kein einziger Artikel relevant ist, sag, dass kein relevanter Artikel gefunden wurde. Hier die Artikel:\n{combined_articles_text}\nErwähne nur die relevanten Artikel und verzichte auf den Hinweis, dass man einen Anwalt beiziehen sollte."
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "Du bist eine Gesetzessumptionsmaschiene. Du beantwortest alle Fragen auf Deutsch."},
                {"role": "user", "content": prompt}
            ]
        )

        if response and response.choices:
            ai_message = response.choices[0].message.content
            st.write(f"Antwort basierend auf der Rechtsstellungsverordnung: {ai_message}")

if __name__ == "__main__":
    main()


