import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from lime.lime_text import LimeTextExplainer
import numpy as np

class DietClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        # Load label mappings
        with open(f"{model_path}/label2id.json", "r") as f:
            self.label2id = json.load(f)
        with open(f"{model_path}/id2label.json", "r") as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}

    def predict(self, text, max_length=128):
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=max_length,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        with torch.no_grad():
            inputs = {
                'input_ids': encoding['input_ids'].to(self.device),
                'attention_mask': encoding['attention_mask'].to(self.device)
            }
            outputs = self.model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            
        return self.id2label[pred_id]
    
    def predict_proba(self, texts, max_length=128):
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {
            'input_ids': encodings['input_ids'].to(self.device),
            'attention_mask': encodings['attention_mask'].to(self.device)
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        return probs.cpu().numpy()


# ----------------------------
# 🚀 Usage + LIME Visualization
# ----------------------------
if __name__ == "__main__":
    classifier = DietClassifier("recipe_bert_model")

    sample_text = "since this is a vegetarian version of biryani,so we do not add to it, instead add paneer pieces"
    prediction = classifier.predict(sample_text)
    print(f"Predicted Category: {prediction}")

    # 🔍 Explain with LIME
    class_names = [classifier.id2label[i] for i in range(len(classifier.id2label))]

    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        sample_text,
        classifier.predict_proba,
        num_features=10,
        num_samples=100  # Reduce from default 5000
    )

    # View in notebook (if using Jupyter)
    # exp.show_in_notebook(text=sample_text)

    # Or save to HTML
    exp.save_to_file("lime_explanation.html")
    print("LIME explanation saved to lime_explanation.html")
