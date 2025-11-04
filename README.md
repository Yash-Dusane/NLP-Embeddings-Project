# NLP-Powered Recipe Analysis System

This project, by Team 61 ALPHA bet from IIIT Hyderabad, is an end-to-end system that analyzes raw recipe text to generate a detailed health chart. The goal is to automatically extract and predict key information from a recipe using text processing and predictive modeling techniques.

---

## Overview

The system performs four main tasks:

1. **Cuisine Classification**: Identifying the regional or cultural origin of the recipe (e.g., Indian, Italian).
2. **Dietary Category Detection**: Classifying the recipe for dietary preferences (e.g., vegan, vegetarian, gluten-free).
3. **Recipe Difficulty Prediction**: Estimating how easy or hard the recipe is to prepare.
4. **Nutritional Value Prediction**: Predicting approximate nutritional content, including calories, fat, protein, and carbohydrates.

---

## Technical Architecture

The project is built on a multi-stage NLP and machine learning pipeline.

### 1. Ingredient Extraction (NER)

This is a critical first step, as ingredient data feeds all other models. Two approaches were used:

* **Rule-Based NER**: A system that uses predefined rules and pattern matching to find ingredient phrases. It extracts quantity, unit, modifiers, and the food name (e.g., "2 tablespoons chopped onion"). This was later enhanced with lemmatization (e.g., "tomatoes" → "tomato") and logic to find the longest possible multi-word ingredient phrases.
* **Machine Learning-Based NER (spaCy)**: A custom NER model trained using spaCy to detect `INGREDIENT` entities in text. This model was fine-tuned from the pre-trained `en_core_web_sm` model by adding the new `INGREDIENT` label.

### 2. Classification Models (Cuisine, Dietary, Difficulty)

* **Model**: These tasks are handled by a BERT (Bidirectional Encoder Representations from Transformers) model, specifically `bert-base-uncased`.
* **Process**: Recipe instructions are tokenized, padded to a fixed length of 128 tokens, and fed into the BERT model. A fully connected layer on top predicts the final class.
* **Training**: The models were trained using the AdamW optimizer and cross-entropy loss. Similar architectures are used for all three classification tasks.

### 3. Nutritional Value Prediction

This is a regression task to predict macro-nutrients. Three different methods were implemented and compared:

1. **Random Forest + One-Hot**: A Random Forest Regressor trained on one-hot encoded vectors of ingredients.
2. **Random Forest + BERT Embeddings**: A Random Forest Regressor trained on BERT-based ingredient embeddings, which are concatenated with quantity information.
3. **Neural Network (PyTorch)**: A custom fully connected neural network (512 → 256 → 128 → output) that uses the mean of all ingredient BERT embeddings as its input.

---

## End-to-End Pipeline

The complete system flow is as follows:

1. **Input**: Raw recipe text.
2. **Ingredient Extraction**: The NER (rule-based + ML-based) models extract all ingredients.
3. **Negation Filtering**: Filters out ingredients that are mentioned but not used (e.g., "serve with or without X").
4. **Embedding**: Text and ingredients are converted into feature embeddings using BERT.
5. **Classification**: The classification models predict cuisine, dietary, and difficulty.
6. **Regression**: The nutrition prediction model estimates macro-nutrient values.
7. **Output**: A health chart summarizing all predictions is generated.

---

## Instructions & Model Access

All trained model files (spaCy NER models, BERT classifiers, etc.) required to run this project are available at the following Google Drive link:

* **Model Files**: [Google Drive Link](https://drive.google.com/drive/folders/1m8ZnHL8bJDzYKE1OsKpmHVREV0xBaoyY?usp=sharing)

To use the system, download the relevant models. You will need to load them into your environment (e.g., spaCy for NER, Transformers/PyTorch for BERT models) and then pass your raw recipe text through the pipeline described above.

---

## Team

* Gopendra Singh - 2022101003
* Yash Nitin Dusane - 2022102078
* **Team 61 ALPHA bet**

---

NOTE : The 3 directories regarding ElMo, RNN_LSTM, and SVD_Word2Vec has the embeddings implementation, independent of Project.
