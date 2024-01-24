import streamlit as st
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Définition des fonctions de votre projet
french_stop_words = [
    'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 
    'il', 'je', 'la', 'le', 'leur', 'lui', 'ma', 'mais', 'me', 'même', 'mes', 'moi', 'mon', 
    'ne', 'nos', 'notre', 'nous', 'on', 'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 
    'se', 'ses', 'son', 'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 
    'vous','les','trs','jai','cest','tout','ils','est','suis','plus','très','sont','fait','cette',
    'depuis','a','à','!','ai','car','faire','j','c','n','alor','alors','d','m','donc','chez','sans'
]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Zéèêëàâäôöûüçîï0-9\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def resume_avis_sumy(text, langue='french', nombre_phrases=2):
    parser = PlaintextParser.from_string(text, Tokenizer(langue))
    summarizer = LsaSummarizer()
    resume = summarizer(parser.document, nombre_phrases)
    return ' '.join([str(phrase) for phrase in resume])

def analyse_et_classifie_sentiments(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.001:
        return 'positif'
    elif polarity < -0.001:
        return 'négatif'
    else:
        return 'neutre'

def plot_wordcloud(corpus):
    text = ' '.join(corpus)
    wordcloud = WordCloud(stopwords=french_stop_words, background_color='white', width=800, height=400).generate(text)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Application Streamlit

def main():
    st.title("Analyse de Texte NLP")

    # Entrée de texte
    texte = st.text_area("Entrez votre avis ici:")

    if st.button("Analyser"):
        if texte:
            # Nettoyage du texte
            texte_nettoye = clean_text(texte)

            # Résumé du texte
            resume = resume_avis_sumy(texte_nettoye)

            # Analyse de sentiment
            sentiment = analyse_et_classifie_sentiments(texte_nettoye)

            # Afficher les résultats
            st.write("Texte Nettoyé:", texte_nettoye)
            st.write("Résumé:", resume)
            st.write("Sentiment:", sentiment)

            # Visualisation des mots fréquents (nuage de mots)
            st.write("Nuage de Mots pour le Texte Saisi:")
            plot_wordcloud([texte_nettoye])

if __name__ == "__main__":
    main()
