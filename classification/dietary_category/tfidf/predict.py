from joblib import load
import numpy as np

class TFIDFRecipeClassifier:
    def __init__(self, model_path):
        self.pipeline = load(model_path)
        self.classes = self.pipeline.classes_  # Get classes directly from pipeline
        
    def predict(self, recipe_text):
        # Ensure input is in correct format
        if isinstance(recipe_text, str):
            recipe_text = [recipe_text]
        return self.pipeline.predict(recipe_text)[0]
    
    def predict_proba(self, recipe_text):
        """Optional: Get probability scores"""
        if isinstance(recipe_text, str):
            recipe_text = [recipe_text]
        return self.pipeline.predict_proba(recipe_text)[0]

if __name__ == "__main__":
    # Initialize classifier
    classifier = TFIDFRecipeClassifier('tfidf_recipe_classifier.joblib')
    
    # Sample prediction
    sample_recipe = """
    Mix rolled oats, almond milk, chia seeds, and honey. 
    Let sit overnight in the refrigerator.
    """
    prediction = classifier.predict(sample_recipe)
    print(f"Predicted Dietary Category: {prediction}")