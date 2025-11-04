import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import joblib
from pathlib import Path

class RecipeClassifier(torch.nn.Module):
    def __init__(self, n_classes=3):
        super(RecipeClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop(pooled_output)
        return self.out(output)

class PredictionPipeline:
    def __init__(self, model_path="recipe_classifier"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path)
        
        # Load assets
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.label_encoder = joblib.load(self.model_path / "label_encoder.joblib")
        self.config = joblib.load(self.model_path / "config.joblib")
        
        # Initialize model
        self.model = RecipeClassifier(len(self.config["classes"]))
        self.model.load_state_dict(torch.load(self.model_path / "bert_classifier.pt", 
                                            map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.config["max_len"],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return encoding["input_ids"], encoding["attention_mask"]
        
    def predict(self, text):
        input_ids, attention_mask = self.preprocess(text)
        
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            outputs = self.model(input_ids, attention_mask)
            _, prediction = torch.max(outputs, dim=1)
            
        return self.label_encoder.inverse_transform(prediction.cpu().numpy())[0]

# Example usage
if __name__ == "__main__":
    pipeline = PredictionPipeline()
    
    sample_text = "Chop vegetables and sauté in pan. Add spices and cook for 10 minutes."
    prediction = pipeline.predict(sample_text)
    print(f"Predicted difficulty: {prediction}")