import torch
from transformers import BertTokenizer
import pickle

# Make sure these match what you used in training
MODEL_PATH = 'bert_recipe_classifier.pt'
ENCODER_PATH = 'label_encoder.pkl'
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load label encoder
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# 2. Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define model class again (should match training)
from torch import nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        out = self.dropout(pooled_output)
        return self.fc(out)

# Recreate and load model
model = BERTClassifier(num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# 3. Define prediction function
def predict_cuisine(recipe_text):
    # Tokenize the input
    encoding = tokenizer(
        recipe_text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        pred_label = torch.argmax(output, dim=1).item()

    cuisine = label_encoder.inverse_transform([pred_label])[0]
    return cuisine

# 4. Try an example
example = "To begin making the Masala Karela Recipe,de-seed the karela and slice.Do not remove the skin as the skin has all the nutrients.Add the karela to the pressure cooker with 3 tablespoon of water, salt and turmeric powder and pressure cook for three whistles.Release the pressure immediately and open the lids.Keep aside.Heat oil in a heavy bottomed pan or a kadhai.Add cumin seeds and let it sizzle.Once the cumin seeds have sizzled, add onions and saute them till it turns golden brown in color.Add the karela, red chilli powder, amchur powder, coriander powder and besan.Stir to combine the masalas into the karela.Drizzle a little extra oil on the top and mix again.Cover the pan and simmer Masala Karela stirring occasionally until everything comes together well.Turn off the heat.Transfer Masala Karela into a serving bowl and serve.Serve Masala Karela along with Panchmel Dal and Phulka for a weekday meal with your family."
predicted = predict_cuisine(example)
print("Predicted cuisine:", predicted)
