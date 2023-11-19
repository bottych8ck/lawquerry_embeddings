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
    paragraphs = data.get(title, [])
    return ' '.join(paragraphs)

def main():
    st.title("Chat-G-TG")

    user_input = st.text_input("Ihre Frage zum Thurgauer Lehrpersonalrecht")
    if st.button("Bearbeiten") and user_input:
        query_vector = get_embeddings(user_input)
        similarities = calculate_similarities(query_vector, article_embeddings)
        sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        top_articles = sorted_articles[:5]
        combined_articles_text = "\n".join([get_article_content(title, law_data) for title, _ in top_articles])

        prompt = f": {user_input}\\n\\Nachfoglend findest Du fünf Gesetzesartikel. Prüfe ob die Artikel relevant sind und beantworte die Frage basierend auf den relevanten Artikel :\\n{combined_articles_text}\\n\\ Verzichte auf folgende Hinweise: Dass man einen Anwalt beiziehen sollte. Dass die anderen Artikel nicht relevant sind."
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


