import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# Configure API URL
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Function to fetch data from the arXiv API
def fetch_arxiv_data(query, max_results=100):
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results
    }
    response = requests.get(ARXIV_API_URL, params=params)
    if response.status_code == 200:
        return response.text
    else:
        print("Error fetching data:", response.status_code)
        return None

# Parse XML response and extract relevant data
def parse_arxiv_response(xml_data):
    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
        category = entry.find("{http://arxiv.org/schemas/atom}primary_category").attrib["term"]
        published_date = entry.find("{http://www.w3.org/2005/Atom}published").text
        papers.append({
            "title": title,
            "summary": summary,
            "category": category,
            "published_date": published_date
        })
    return papers

# Generate embeddings using Sentence-BERT
def generate_embeddings(papers, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [paper["summary"] for paper in papers]
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

# Perform K-Means clustering
def cluster_papers(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

# Perform LDA topic modeling
def topic_modeling(papers, num_topics=5):
    texts = [paper["summary"].split() for paper in papers]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_words=5)
    return topics

# Visualize cluster distribution
def visualize_clusters(labels, embeddings):
    reduced_embeddings = embeddings[:, :2]  # Assuming embeddings are reduced to 2D
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis")
    plt.title("Cluster Distribution")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.colorbar()
    plt.show()

# Main script
if __name__ == "__main__":
    query = "machine learning AND healthcare"  # Example query
    xml_data = fetch_arxiv_data(query)
    
    if xml_data:
        papers = parse_arxiv_response(xml_data)
        print(f"Retrieved {len(papers)} papers.")
        
        # Generate embeddings
        embeddings = generate_embeddings(papers)
        
        # Perform clustering
        labels = cluster_papers(embeddings)
        
        # Perform topic modeling
        topics = topic_modeling(papers)
        print("Identified Topics:")
        for topic in topics:
            print(topic)
        
        # Visualize clusters
        visualize_clusters(labels, embeddings)
