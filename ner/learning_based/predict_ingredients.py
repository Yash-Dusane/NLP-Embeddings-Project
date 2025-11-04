import spacy

# Load the trained model
model_path = "custom_ner_1"
# model_path="custom_ner_2"
nlp = spacy.load(model_path)

# Input text
text = "to make the dosa recipe , first add some coconut oil into the pan, then het it for sometime then add rice and potato. then serve it nicely"

# Process the text
doc = nlp(text)

# Use a set to store unique ingredients
unique_ingredients = set()

for ent in doc.ents:
    if ent.label_ == "INGREDIENT":
        # Normalize text (e.g. lowercase and strip whitespace)
        normalized = ent.text.strip().lower()
        unique_ingredients.add(normalized)

# Print sorted ingredients
print("Ingredients found:")
for ingredient in sorted(unique_ingredients):
    print(f"- {ingredient}")
