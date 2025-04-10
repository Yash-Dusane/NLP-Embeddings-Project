The Assignment contains a `main.ipynb` file containing
- Data Preprocessing
- SVD
- CBOW
- SkipGram

There are independent word embedding `.py` files in `main.ipynb`

To execute the word embedding files, simply run the following command:
```
python3 filename.py
```
OUTPUT : `.pt` file with trained model being saved.


GOOGLE DRIVE LINK to `.pt` files : https://drive.google.com/drive/folders/1TfAiZYZZCbYit1OT1f1tj_nVF8R-Q7VU?usp=sharing


---------------------

Further, there is a `wordsim.py` file which is to be run to generate the Spearman correlation, run the following command:
```
python3 wordsim.py <arg1>
```
where `<arg1>` is the path to the embedding file.

`<arg1>` : `<embedding_path>.pt` 

----------------------

OUTPUT : The Spearman correlation value, and code output i.e. correlation of Words of `wordsim353crowd.csv`
with the words by `.pt` file in a `.csv` file.

-----------------------


