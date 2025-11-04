import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

class DietClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        # Load label mappings
        with open(f"{model_path}/label2id.json", "r") as f:
            self.label2id = json.load(f)
        self.id2label = {int(k):v for k,v in json.load(open(f"{model_path}/id2label.json")).items()}

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

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = DietClassifier("recipe_bert_model")
    
    # Sample prediction
    sample_recipe =  "this is a recipe that uses jackfruit which almost tastes like chicken but is fully vegan"
    
    prediction = classifier.predict(sample_recipe)
    print(f"Predicted Dietary Category: {prediction}")