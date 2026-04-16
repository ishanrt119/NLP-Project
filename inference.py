import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration
import numpy as np

class BiasInference:
    def __init__(self, classifier_path, rewriter_path):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Classifier
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.classifier = self._load_classifier(classifier_path)
        self.classifier.eval()
        
        # Load Rewriter
        self.t5_tokenizer = T5Tokenizer.from_pretrained(rewriter_path)
        self.rewriter = T5ForConditionalGeneration.from_pretrained(rewriter_path)
        self.rewriter.to(self.device)
        self.rewriter.eval()
        
        self.label_cols = ['Confirmation Bias', 'Overconfidence Bias', 'Anchoring Bias']

    def _load_classifier(self, path):
        # Local definition of classifier for loading
        class BiasClassifier(nn.Module):
            def __init__(self, n_classes):
                super(BiasClassifier, self).__init__()
                self.bert = BertModel.from_pretrained('bert-base-uncased')
                self.drop = nn.Dropout(p=0.3)
                self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                return self.out(outputs.pooler_output)
        
        model = BiasClassifier(len(self.label_cols))
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        return model

    def analyze(self, text):
        # 1. Classification
        encoding = self.bert_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        detected_biases = []
        for i, prob in enumerate(probs):
            if prob > 0.5:
                detected_biases.append({
                    "label": self.label_cols[i],
                    "confidence": float(prob)
                })
        
        # 2. Rewriting (only if bias detected)
        rewritten_text = text
        if detected_biases:
            input_text = "rewrite to remove bias: " + text
            input_ids = self.t5_tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.rewriter.generate(input_ids, max_length=128)
                rewritten_text = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 3. Simple Keyword Highlighting (Dynamic)
        bias_keywords = {
            "Confirmation Bias": ["ignore", "believe", "only read", "proven wrong", "mind is made up"],
            "Overconfidence Bias": ["100%", "no doubt", "guaranteed", "flawless", "impossible", "know for a fact"],
            "Anchoring Bias": ["original price", "first estimate", "starting point", "sticking to", "anchor"]
        }
        
        highlighted_words = []
        for bias in detected_biases:
            for word in bias_keywords.get(bias["label"], []):
                if word.lower() in text.lower():
                    highlighted_words.append(word)
        
        return {
            "detected_biases": detected_biases,
            "rewritten_text": rewritten_text,
            "highlighted_words": list(set(highlighted_words)),
            "is_biased": len(detected_biases) > 0
        }
