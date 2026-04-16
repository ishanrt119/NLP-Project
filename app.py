import streamlit as st
import pandas as pd
from inference import BiasInference
import os

# Page configuration
st.set_page_config(
    page_title="AI Cognitive Bias Detector",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for modern look
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
        color: #e0e0e0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #1e222a;
        color: white;
    }
    .highlight {
        background-color: #3e4451;
        padding: 2px 4px;
        border-radius: 4px;
        color: #ffcc00;
        font-weight: bold;
    }
    .bias-card {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #1a1c23;
        border: 1px solid #30363d;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Header
st.title("🧠 Cognitive Bias Detection & Intelligent Rewriting")
st.markdown("---")

# Load Inference Engine
@st.cache_resource
def load_engine():
    classifier_path = "models/bias_classifier.bin"
    rewriter_path = "models/t5_rewriter"
    if not os.path.exists(classifier_path) or not os.path.exists(rewriter_path):
        return None
    return BiasInference(classifier_path, rewriter_path)

engine = load_engine()

if engine is None:
    st.warning("⚠️ Models are still training or not found in `models/` directory. Please run training scripts first.")
else:
    # Sidebar for Tone Control
    st.sidebar.title("Settings")
    tone = st.sidebar.selectbox("Tone Control", ["Neutral", "Professional"])
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Text")
        user_input = st.text_area("Enter a sentence to analyze for cognitive bias:", height=150, placeholder="e.g., I am 100% certain this investment will double by next month.")
        
        analyze_btn = st.button("Analyze & Rewrite")

    # Results section
    if analyze_btn and user_input:
        results = engine.analyze(user_input)
        
        st.markdown("---")
        st.subheader("Analysis Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("#### Detected Biases")
            if not results["is_biased"]:
                st.success("No cognitive bias detected! The statement appears logical and neutral.")
            else:
                for bias in results["detected_biases"]:
                    st.error(f"**{bias['label']}** (Confidence: {bias['confidence']:.2%})")
                
                st.markdown("**Highlighted Keywords:**")
                if results["highlighted_words"]:
                    highlighted_txt = user_input
                    for word in results["highlighted_words"]:
                        highlighted_txt = highlighted_txt.replace(word, f'<span class="highlight">{word}</span>')
                    st.markdown(f'<p style="font-size: 1.2rem;">{highlighted_txt}</p>', unsafe_allow_html=True)
                else:
                    st.write("No specific keywords highlighted, but bias structure detected.")

        with res_col2:
            st.markdown("#### Logical Rewrite")
            if results["is_biased"]:
                final_rewrite = results["rewritten_text"]
                if tone == "Professional":
                    final_rewrite = "Upon critical review, " + final_rewrite
                
                st.info(f"**Bias-Free Version:**\n\n{final_rewrite}")
                st.markdown("> **Explanation:** The rewritten version uses probabilistic language (e.g., 'likely', 'suggests') rather than absolute certainty, reducing overconfidence and anchoring effects.")
            else:
                st.info("Input was already neutral. No rewriting necessary.")

    # Footer
    st.markdown("---")
    st.caption("Advanced NLP Project | Built with BERT & T5 Transformers")
