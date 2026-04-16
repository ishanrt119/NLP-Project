# AI Cognitive Bias Detector & Text Rewriter

This project is an advanced NLP system designed to detect cognitive biases (Confirmation, Overconfidence, and Anchoring) in text and rewrite them into neutral, rational, and evidence-based statements using Transformer models.

## 🚀 Features
- **Multi-Label Detection**: Uses **BERT** (`bert-base-uncased`) to identify multiple biases in a single sentence.
- **Intelligent Rewriting**: Uses **T5** (`t5-small`) to generate neutral alternatives.
- **Modern Dashboard**: Built with **Streamlit** for real-time analysis and visualization.
- **Hardware Optimized**: Automatically uses **Apple Silicon (MPS)** or **CUDA** if available.

## 🛠 Tech Stack
- **Models**: BERT (Encoder), T5 (Encoder-Decoder)
- **Frameworks**: PyTorch, HuggingFace Transformers
- **UI**: Streamlit
- **Preprocessing**: Scikit-learn, Pandas

## 📁 Project Structure
```text
├── app.py                  # Streamlit UI
├── inference.py            # Combined model logic
├── train_classifier.py     # BERT training script
├── train_rewriter.py       # T5 training script
├── data_preprocessing.py   # Dataset handling utilities
├── data_generator.py       # Synthetic data generation
├── requirements.txt        # Project dependencies
└── models/                 # Saved model weights
```

## 📋 How to Run Locally

The repository excludes trained models and the raw dataset to keep the repository lightweight. You can easily regenerate them locally using the scripts provided.

### 1. Install Dependencies
```bash
python3 -m pip install -r requirements.txt
python3 -m pip install sentencepiece  # Required for T5
```

### 2. Generate Dataset
This creates `bias_dataset.csv` with 5,000 synthetic samples.
```bash
python3 data_generator.py
```

### 3. Build & Train Models
Run these scripts to fine-tune the BERT and T5 models on the generated data. This takes ~5-10 minutes on Apple Silicon.
```bash
python3 train_classifier.py
python3 train_rewriter.py
```

### 4. Launch the App
Once the `models/` directory is populated, launch the dashboard:
```bash
streamlit run app.py
```

## 💡 Example Inputs
- **Overconfidence**: "I am 100% certain that this stock will triple in value by next week."
- **Confirmation**: "I don't care what the studies say; I know vaccines are harmful because I've always believed it."
- **Anchoring**: "The original price was $5000, so $2500 is definitely a bargain for this laptop."

## 📄 License
This project is built for educational purposes.
