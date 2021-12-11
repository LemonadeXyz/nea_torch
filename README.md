# Neural Essay Assessor: A Python3 and PyTorch Implementation

This is a new implementation of Neural Essay Assessor(NEA) reproduced with ***Python3*** and the very popular deep learning framework ***PyTorch***!

Click [here](https://github.com/nusnlp/nea) to learn about the old NEA, a classic neural model for automatic scoring of English essays.

## Set up
pass

## Data

Regarding the related operations of data downloading and preprocessing, the method and channels of this project are exactly the same as those of the old NEA, as described in the original repo:

> We have used 5-fold cross validation on ASAP dataset to evaluate our system. This dataset (training_set_rel3.tsv) can be downloaded from [here](https://www.kaggle.com/c/asap-aes/data). After downloading the file, put it in the [data](https://github.com/LemonadeXyz/nea_torch/tree/main/data) directory and create training, development and test data using preprocess_asap.py script:
 ```
cd data
python preprocess_asap.py -i training_set_rel3.tsv
```
Or you can directly use the preprocessed [data](https://github.com/nusnlp/nea/tree/master/data) to run the code (this is what I did actually).


