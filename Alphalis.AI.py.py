import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import torch


user_login = st.text_input("Enter your login: ")
user_password = st.text_input("Enter your password: ", type="password")

# 1. Page Configuration
st.set_page_config(page_title="AlphalisAI", layout="wide")

# 2. Injecting Custom CSS for a Dark, Minimalist, and Soft Aesthetic
st.markdown("""
    <style>
    /* Dark, dreamy background */
    .stApp {
        background-color: #141416;
        color: #e0e0e0;
    }
    /* Soft, vintage typography for headers */
    h1, h2, h3 {
        color: #d8c8b8; 
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 300;
        letter-spacing: 1px;
    }
    /* Minimalist card containers */
    div.css-1r6slb0, div.css-12oz5g7 {
        background-color: #1e1e20;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2a2a2d;
    }
    /* Soft interactive buttons */
    .stButton>button {
        background-color: #242426;
        color: #d8c8b8;
        border: 1px solid #404044;
        border-radius: 8px;
        transition: 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #d8c8b8;
        color: #141416;
        border: 1px solid #d8c8b8;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Application Header
st.title("AlphalisAI")
st.markdown("Map complex ecological rules by fusing audio embeddings with spatial telemetry.")
st.divider()

# 4. Sidebar for Model Parameters
with st.sidebar:
    st.header("Pipeline Parameters")
    confidence_threshold = st.slider("Embedding Confidence Threshold", 0.0, 1.0, 0.75)
    st.write("Graph Architecture Settings:")
    edge_weight = st.selectbox("Edge Weight Metric", ["Physical Distance", "Social Hierarchy", "Acoustic Overlap"])

# 5. Main Layout: Upload & Processing
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Data Ingestion")
    uploaded_audio = st.file_uploader("Upload Acoustic Data (.wav)", type=['wav'])
    uploaded_telemetry = st.file_uploader("Upload Spatial Telemetry (.csv)", type=['csv'])
    
    analyze_button = st.button("Generate Ecological Translation")

with col2:
    st.subheader("System Status")
    status_text = st.empty()
    if not uploaded_audio:
        status_text.info("Awaiting data streams...")

# 6. Mock Inference & Visualization Logic
if analyze_button and uploaded_audio:
    with col2:
        status_text.warning("Extracting Audio Embeddings via AVES...")
        time.sleep(1.5) # Mocking PyTorch processing time
        status_text.warning("Constructing Social-Spatial Graph...")
        time.sleep(1.5)
        status_text.success("Translation Complete.")
    
    st.divider()
    
    # Results Section
    st.subheader("Predicted Behavioral Translation")
    st.write("**Ecological Intent:** Coordinated directional shift (Foraging behavior triggered by acoustic pulse).")
    
    # Generate a mock visualization of the Spatiotemporal Graph
    st.write("### Spatial Network Impact")
    
    # Mocking the Graph Neural Network output using NetworkX
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#C4C4F8") # Match the dark UI background
    ax.set_facecolor("#EBBEF5")
    
    G = nx.erdos_renyi_graph(n=8, p=0.4, seed=42)
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='#d8c8b8', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#F1A0A0", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="#141416", font_family="sans-serif", ax=ax)
    
    ax.axis('off')
    st.pyplot(fig)
