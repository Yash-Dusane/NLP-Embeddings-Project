
import pandas as pd
import re
import json
from tqdm import tqdm

def extract_ingredient_variants(ingredient):
    """
    Extract main and bracketed ingredient parts.
    e.g., 'gram flour (besan)' → ['gram flour', 'besan']
    """
    ingredient = ingredient.strip()
    variants = []

    # Match content like "text (text)"
    match = re.match(r'^(.*?)\s*\((.*?)\)$', ingredient)
    if match:
        outer = match.group(1).strip()
        inner = match.group(2).strip()
        if outer:
            variants.append(outer)
        if inner:
            variants.append(inner)
    else:
        if ingredient:
            variants.append(ingredient.strip())
    
    return variants

# Load your dataset
df = pd.read_csv("data/Cleaned_Indian_Food_Dataset.csv")  # Replace with your filename

training_data = []

for idx, row in tqdm(df.iterrows()):
    recipe_text = str(row["TranslatedInstructions"])
    recipe_text = str(row["TranslatedInstructions"]).replace('\\n', ' ').replace('\n', ' ')
    recipe_text_lower = recipe_text.lower()
    raw_ingredients = str(row["Cleaned-Ingredients"]).split(",")  # Assuming comma-separated

    entities = []
    for raw_ing in raw_ingredients:
        ing_variants = extract_ingredient_variants(raw_ing)
        for ing in ing_variants:
            ing = ing.lower()
            # Find all non-overlapping matches
            for match in re.finditer(r'\b' + re.escape(ing) + r'\b', recipe_text_lower):
                start, end = match.start(), match.end()
                entities.append((start, end, "INGREDIENT"))
                # print(start,end," : ",ing)

    if entities:
        training_data.append((recipe_text, {"entities": entities}))

with open("train_data.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

print("✅ Training data saved to train_data.json")