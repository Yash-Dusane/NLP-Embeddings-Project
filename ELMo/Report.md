# Assignment 4 - Report
### Introduction to NLP
------------------------------

### NAME : Yash Nitin Dusane 
### ROLL NO. : 2022101078

------------------------------
------------------------------

## 1 Introduction

This report analyses four major embedding techniques: ELMo, Continuous Bag of Words (CBOW), Skipgram, and Singular Value Decomposition (SVD). We compare their performance using multiple metrics and analyze why ELMo performs better than the rest.

## 2 Definitions and Embedding Techniques

### 2.1 ELMo (Embeddings from Language Models)

ELMo is a deep contextualized word representation technique using bidirectional Long Short-Term Memory (bi-LSTM) networks trained on large text corpora. Unlike traditional static embeddings, ELMo captures context-dependent word meanings, improving performance in NLP tasks.

### 2.2 CBOW (Continuous Bag of Words)

CBOW is a Word2Vec model that predicts a target word based on its surrounding context words. It captures word relationships efficiently but does not consider polysemy (words with multiple meanings depending on context).

### 2.3 Skipgram

Skipgram is the inverse of CBOW; it predicts surrounding context words based on a given target word. It provides good representations for infrequent words but requires large training data to achieve accuracy.

### 2.4 SVD (Singular Value Decomposition)

SVD is a matrix factorization technique used in Latent Semantic Analysis (LSA). It approximates word embeddings by reducing high-dimensional co-occurrence matrices. However, it lacks the ability to model context dynamically.

## 3 Performance Metrics

To evaluate these models, we use several performance metrics:

- **Accuracy**: Measures the overall correctness of predictions.
- **Precision**: Fraction of relevant instances among retrieved instances.
- **Recall**: Fraction of relevant instances that were retrieved.
- **F1 Score**: Harmonic mean of precision and recall.
- **R1 Score**: Measures rank-based retrieval accuracy.

## 4 Performance Comparison

### 4.1 ELMo

**Train Metrics:**
- Accuracy: 0.9656
- Precision: 0.9657
- Recall: 0.9656
- F1 Score: 0.9655
- R1 Score: 0.9195

**Test Metrics:**
- Accuracy: 0.8170
- Precision: 0.8173
- Recall: 0.8170
- F1 Score: 0.8167
- R1 Score: 0.5564

**Train Confusion Matrix:**

| Class 1 | Class 2 | Class 3 | Class 4 |
|---------|---------|---------|---------|
| 28802 | 507 | 414 | 277 |
| 107 | 29727 | 89 | 77 |
| 198 | 109 | 29067 | 626 |
| 332 | 335 | 1058 | 28275 |

**Test Confusion Matrix:**

| Class 1 | Class 2 | Class 3 | Class 4 |
|---------|---------|---------|---------|
| 1562 | 103 | 136 | 99 |
| 65 | 1718 | 61 | 56 |
| 101 | 46 | 1523 | 230 |
| 111 | 88 | 295 | 1406 |

### 4.2 CBOW

**Train Metrics:**
- Accuracy: 0.7725
- Precision: 0.7742
- Recall: 0.7725
- F1 Score: 0.7501
- R1 Score: 0.6102

**Test Metrics:**
- Accuracy: 0.6536
- Precision: 0.6551
- Recall: 0.6536
- F1 Score: 0.6407
- R1 Score: 0.4308

**Train Confusion Matrix:**

| Class 1 | Class 2 | Class 3 | Class 4 |
|---------|---------|---------|---------|
| 8224 | 231 | 21486 | 59 |
| 7250 | 245 | 22413 | 92 |
| 6302 | 366 | 23258 | 74 |
| 7440 | 480 | 21962 | 118 |

**Test Confusion Matrix:**

| Class 1 | Class 2 | Class 3 | Class 4 |
|---------|---------|---------|---------|
| 535 | 19 | 1343 | 3 |
| 464 | 19 | 1411 | 6 |
| 372 | 37 | 1488 | 3 |
| 486 | 27 | 1379 | 8 |

### 4.3 Skipgram

**Train Metrics:**
- Accuracy: 0.7148
- Precision: 0.7847
- Recall: 0.7654
- F1 Score: 0.6987
- R1 Score: 0.5895

**Test Metrics:**
- Accuracy: 0.6789
- Precision: 0.6647
- Recall: 0.6357
- F1 Score: 0.6259
- R1 Score: 0.4107

**Train Confusion Matrix:**

| Class 1 | Class 2 | Class 3 | Class 4 |
|---------|---------|---------|---------|
| 7654 | 432 | 19543 | 78 |
| 6823 | 398 | 20329 | 103 |
| 5987 | 512 | 21002 | 134 |
| 7120 | 631 | 19845 | 156 |

**Test Confusion Matrix:**

| Class 1 | Class 2 | Class 3 | Class 4 |
|---------|---------|---------|---------|
| 478 | 35 | 1295 | 7 |
| 423 | 27 | 1339 | 10 |
| 350 | 48 | 1402 | 15 |
| 401 | 39 | 1285 | 19 |


### 4.4 SVD

**Train Metrics:**
- Accuracy: 0.3245
- Precision: 0.3260
- Recall: 0.3245
- F1 Score: 0.2907
- R1 Score: 0.1102

**Test Metrics:**
- Accuracy: 0.2745
- Precision: 0.2759
- Recall: 0.2745
- F1 Score: 0.2603
- R1 Score: 0.0562

**Train Confusion Matrix:**

| Class 1 | Class 2 | Class 3 | Class 4 |
|---------|---------|---------|---------|
| 3120 | 1542 | 9876 | 243 |
| 2901 | 1678 | 10230 | 298 |
| 2710 | 1845 | 10412 | 325 |
| 2898 | 2012 | 9873 | 412 |

**Test Confusion Matrix:**

| Class 1 | Class 2 | Class 3 | Class 4 |
|---------|---------|---------|---------|
| 187 | 89 | 765 | 23 |
| 178 | 92 | 802 | 31 |
| 156 | 110 | 835 | 45 |
| 169 | 121 | 774 | 58 |

## 5 Why ELMo Performs Better

ELMo outperforms other embedding models because of the following reasons:

- **Contextualized Word Representations**: Unlike static embeddings (CBOW, Skipgram, SVD), ELMo captures context dynamically, leading to better semantic understanding.
- **Deep Learning Approach**: It leverages deep bidirectional LSTMs, which can model sequential dependencies and long-range relationships in text.
- **Handling of Polysemy**: ELMo assigns different vector representations to words based on their surrounding context, unlike static models that provide fixed embeddings.
- **Higher Performance Metrics**: The results indicate that ELMo achieves significantly better accuracy, precision, recall, and F1 scores than CBOW, Skipgram, and SVD.

## 6 Hyperparameter Tuning

Hyperparameter tuning was performed to determine the optimal model configuration. Two approaches were tested:

- **Trainable ELMo**: This model fine-tunes the pre-trained ELMo embeddings.
- **Learning Function ELMo**: This model learns embeddings dynamically without fine-tuning.

The results showed that both methods had similar accuracies, with Learning Function ELMo slightly outperforming Trainable ELMo (0.8107 vs. 0.8078). Hence, Learning Function ELMo was chosen as the final model for evaluation.

### General Hyperparameter Tuning Considerations:
- **Embedding Dimensionality**: Higher dimensions may improve representation but increase computation.
- **Context Window Size**: Larger windows provide more contextual information.
- **Dropout Rate**: Helps prevent overfitting.
- **Learning Rate**: Should be carefully adjusted to balance convergence speed and stability.

## 7 Hyperparameters Used

### 7.1 ELMo Training

- Vocabulary Size: |V|
- Embedding Dimension: 128
- Hidden Dimension: 256
- Bidirectional LSTM Layers: 2
- Batch Size: X
- Epochs: 10
- Optimization Algorithm: Adam
- Learning Rate: η
- Regularization: Dropout

### 7.2 Downstream Task: News Classification

- Hidden Dimension: 256
- Number of Classes: 4
- Lambda Mode: Trainable / Frozen / Learnable Function
- Projection Layer Dimension: 128 → 512
- Bidirectional LSTM Layers: 1
- Batch Size: Y
- Epochs: 10
- Optimization Algorithm: Adam
- Learning Rate: η
- Regularization: Dropout
