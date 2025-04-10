# Assignment 2

The assignment contains the perplexity score files, a ipynb file for the overall classes and analysis.


Get the pt files for pretrained models from the link :
https://drive.google.com/drive/folders/1IQXAQcN6ssdbAPfcqFUhI5MgOREco0yj?usp=sharing


Further there is a file generator.py as per the question requirements, which is in the asked format.


---------

## Experimental Setup :

2 Datasets being given for the veriication

### Hyperparameters
 • Embedding Dimension:  100 or 200
• Hidden Layer Dimension: 50
• Optimizer: Adam 
• Loss Function: Cross-Entropy 
• Training Epochs: 10 

( The parameters are varied accordingly to get better results )

1.3 Perplexity Calculation

 Perplexity measures how well a model predicts unseen words. 
 
 - Lower values indicate better generalization. 



 ## Results and Analysis :

 2.1 Perplexity Scores for Pride and Prejudice

 | Model         | Train Perplexity | Test Perplexity |
|--------------|----------------|---------------|
| FFNN (n=5)  | 53.42176636527214       | 1922.0938545597317          |
| FFNN (n=3)  | 60.482822350895795         | 1013.4529580449184             | 
| RNN         | 56.92496411546799      | 85.06823490394767   | 
| LSTM        | 58.87300052804184      | 140.42123108371055   | 

For Ulysses :

 | Model         | Train Perplexity | Test Perplexity |
|--------------|----------------|---------------|
| FFNN (n=5)  | 450.4624248806277      | 2305.1073157588503          |
| FFNN (n=3)  | 301.7608527522701        | 1304.991477433963             | 
| RNN         | 141.30311310814326     | 143.47683560335636   | 
| LSTM        | 89.35458035385253     | 931.8270328033876  | 



### Comparing with the results of Assignment 1 Smoothing Probability Techniques :

- P&P :

( train, test )

Laplace : 

N=3 : 762, 1292
N=5 : 468, 796

Good Turing :

N=3 : 11400, 17184
N=5 : 33636, 52759

Linear Interpolation :

N=3 : 2, 32
N=5 : 1.3, 18


- Ulysses

( train, test )

Laplace : 

N=3 : 1669, 2846
N=5 : 786, 1361

Good Turing :

N=3 : 882287, 909123
N=5 : 2501482, 2544673

Linear Interpolation :

N=3 : 2, 95
N=5 : 1.16, 40



So, seeing the values of Smoothening, all of them give different values, but when compared with the Neural Networks, they have almost similar perplexiies for P&P and also the test perplexity vary according to how good the model is.

We have similar trend for Ulysses corpus also.


Overall Neural Networks is a better implmentation than Smoothening Techniques due to stable results.


Neural language models, such as FFNN, RNN, and LSTM, outperform traditional smoothing 
techniques like Laplace smoothing, Good-Turing smoothing, and linear 
interpolation because they learn complex patterns in language. Traditional methods rely on fixed 
probability adjustments and cannot capture deep contextual relationships within sequences. Neural 
models exploit distributed word representations (embeddings) and hierarchical feature learning to 
dynamically adapt to varying sentenc



## Performance Comparison :

### FFNN :

- The train perplexity here doent vary much DUE TO SUITABLE HYPERPARAMETERS. But we get relatively higher test perplexity values.

-  FFNN relies on fixed-length n-grams, making it incapable of handling varied-length 
sentences or generalizing to unseen contexts

- With increase in N, the model overall gives unstable results in test perplexity, and tries to overfit the training sentences.


### RNN :

- RNN is far better than FFNN, acheiving a significantly lower test perplexity
- Further train perplexity also is stable and at the same range
• The vanishing gradient problem limits its ability to process long sequences effectively. 
• Performs better than FFNN but falls short of LSTM.
(Not seen here)

### LSTM :

- LSTM is the best model, and for longer sequences would give the best results. (It couldnt be shown here).
- Gating mechanisms allow LSTM to retain long-range dependencies, making it the best choice for sequential text prediction. 
- Better Generalisation




## Observations :


### Effect of N-Gram Size on FFNN Performance


 The n-gram size significantly impacts FFNN performance:
 
As mentioned earlier, with increase in N, we could see OVERFITTING

 1. Overfitting Increases with n: Larger n-grams improve training performance but significantly degrade test generalization. 
2. Data Sparsity Problem: A higher n increases the number of unique n-grams , making it difficult to learn meaningful probabilities from limited training data. 
3. Fixed Context Limitation: Unlike RNNs and LSTMs, FFNN lacks sequential memory, restricting its ability to generalize beyond fixed-length n-grams. 


### What in case of Longer Sentences ?

 LSTM and RNN  ( LSTM is better ) performs best for longer sentences due to its ability to retain long-range dependencies using gating mechanisms. 
 
- FFNN fails for longer sentences due to its fixed n-gram input size. 
• RNN suffers from the vanishing gradient problem, limiting its ability to model long sequences effectively. 
• LSTM's memory cell structure allows it to learn dependencies over longer spans, making it the best choice. 


Generation Example :


Input sentence: I have 
been: 0.0715
to: 0.0596
a: 0.0391
LSTM Predictions: [('been', 0.07151152938604355), ('to', 0.05959521606564522), ('a', 0.039139967411756516)]


