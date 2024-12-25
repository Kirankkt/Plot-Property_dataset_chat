import os
import pandas as pd
import numpy as np
import faiss
import streamlit as st
import google.generativeai as gemini
from fuzzywuzzy import process

# Set your Gemini API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyBy791VYFuQjFIkCTV_ELBkGKIsv17wH_M'

# Initialize Gemini API
gemini.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Load datasets
property_df = pd.read_csv('https://github.com/Kirankkt/Plot-Property_dataset_chat/blob/main/Updated_Cleaned_Dataset%20(1).csv')
plot_df = pd.read_csv('https://github.com/Kirankkt/Plot-Property_dataset_chat/blob/main/standardized_locations_dataset.csv')

# Function to preprocess data
def preprocess_data(df, text_column):
    df.fillna('', inplace=True)
    return df[text_column].tolist()

# Extract textual data for embeddings
property_texts = preprocess_data(property_df, 'Plot__DESC')
plot_texts = preprocess_data(plot_df, 'Location')

# Function to generate embeddings
def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        response = gemini.generate_embeddings(text=text)
        embeddings.append(response['embedding'])
    return embeddings

# Generate embeddings for both datasets
property_embeddings = generate_embeddings(property_texts)
plot_embeddings = generate_embeddings(plot_texts)

# Combine embeddings and create a FAISS index
all_embeddings = property_embeddings + plot_embeddings
embedding_matrix = np.array(all_embeddings).astype('float32')

index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# Create a mapping to identify the source of each embedding
embedding_sources = ['property'] * len(property_embeddings) + ['plot'] * len(plot_embeddings)

# Fuzzy matching function
def fuzzy_match_location(query, locations):
    match, score = process.extractOne(query, locations)
    if score > 80:  # Threshold for fuzzy matching
        return match
    return None

# Streamlit user interface
st.title('RAG Chatbot')

user_input = st.text_input('Ask a question:')

if user_input:
    # Extract location names from both datasets
    all_locations = list(plot_df['Location'].unique()) + list(property_df['Standardized_Location_Name'].unique())

    # Check if the user query contains a location
    location_in_query = fuzzy_match_location(user_input, all_locations)
    if location_in_query:
        st.write(f"Interpreting query with location: {location_in_query}")

    # Generate embedding for user query
    query_embedding = gemini.generate_embeddings(text=user_input)['embedding']

    # Search for similar embeddings in FAISS
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=5)

    # Retrieve corresponding data entries and their sources
    retrieved_texts = []
    for idx in I[0]:
        source = embedding_sources[idx]
        if source == 'property':
            retrieved_texts.append(property_texts[idx])
        else:
            retrieved_texts.append(plot_texts[idx])

    # Check if relevant data is found
    if retrieved_texts:
        context = ' '.join(retrieved_texts)
        response = gemini.generate_text(prompt=f"Context: {context}\n\nQuestion: {user_input}\nAnswer:")
        st.write(response['generated_text'])
    else:
        st.write("Sorry, this information is not available in the datasets.")
