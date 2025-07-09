# app.py

import streamlit as st
st.set_page_config(page_title="Lecture Summary Finder", layout="centered")
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import re
import html

# ------- Clean text -------
def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ------- Load preprocessed data & models -------
@st.cache_resource
def load_models_and_data():
    df = pd.read_csv("Clustered_Summaries.csv")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    summaries = df['Cleaned_Summary'].tolist()
    summary_embeddings = embed(summaries).numpy()

    kmeans = KMeans(n_clusters=df['Lecture_Label'].nunique(), random_state=40)
    kmeans.fit(summary_embeddings)

    centroids = kmeans.cluster_centers_

    cluster_nametags = {}
    for label in sorted(df['Lecture_Label'].unique()):
        topic = df[df['Lecture_Label'] == label]['Lecture_Name'].iloc[0]
        cluster_nametags[label] = topic

    return df, embed, kmeans, centroids, cluster_nametags

df, embed, kmeans, cluster_centroids, cluster_nametags = load_models_and_data()

# ------- Predict function -------
def predict_cluster(input_summary, top_n=3):
    cleaned = clean_text(input_summary.lower().strip())
    embedding = embed([cleaned]).numpy()
    label = kmeans.predict(embedding)[0]
    cluster_name = cluster_nametags.get(label, f"Topic {label}")

    confidence = cosine_similarity(embedding, [cluster_centroids[label]])[0][0]

    cluster_data = df[df['Lecture_Label'] == label].copy()
    cluster_embeddings = embed(cluster_data['Cleaned_Summary'].tolist()).numpy()
    similarities = cosine_similarity([cluster_centroids[label]], cluster_embeddings)[0]
    cluster_data['SimilarityToCentroid'] = similarities
    top_summaries = cluster_data.sort_values(by='SimilarityToCentroid', ascending=False).head(top_n)

    return label, cluster_name, confidence, top_summaries

# ------- Streamlit UI -------

st.title("🎓 Lecture Summary Finder")
st.markdown("Enter any sentence or keywords to get the most relevant lecture and summaries.")

user_input = st.text_input("🔍 Enter keywords:")

if user_input:
    with st.spinner("Analyzing..."):
        label, name, confidence, top_summaries = predict_cluster(user_input)

    st.success(f"📘 Predicted Lecture Cluster: **{name}** (Cluster {label})")
    st.info(f"🔗 Similarity to Cluster Centroid: `{confidence:.4f}`")

    st.subheader("📝 Top 3 Representative Summaries")
    for i, row in top_summaries.iterrows():
        st.markdown(f"**•** {row['Session_Summary']}")
