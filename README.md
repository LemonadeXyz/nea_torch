# Neural Essay Assessor: A Python3 and PyTorch Implementation

This is a new implementation of Neural Essay Assessor(NEA) reproduced with ***Python3*** and the very popular deep learning framework ***PyTorch***!

Click [here](https://github.com/nusnlp/nea) to learn about the old NEA, a classic neural model for automatic scoring of English essays.

### Enviroment

- Python 3.6+

### Packages

- numpy 1.19.5
- scipy 1.5.2
- sci-kit-learn 0.24.1
- pytorch 1.7.1

***NOTE：*** NEA-torch is now only tested on **CPU**! GPU-related operations are taken into account when building the project, but successful execution cannot be guaranteed.

## Set up

- Configure the environment
- Prepare data
- Run `main_nea.py`


## Data

Regarding the related operations of data downloading and preprocessing, the method and channels of this project are exactly the same as those of the old NEA, as described in the original repo:

> We have used 5-fold cross validation on ASAP dataset to evaluate our system. This dataset (training_set_rel3.tsv) can be downloaded from [here](https://www.kaggle.com/c/asap-aes/data). After downloading the file, put it in the [data](https://github.com/LemonadeXyz/nea_torch/tree/main/data) directory and create training, development and test data using preprocess_asap.py script:
 ```bash
cd data
python preprocess_asap.py -i training_set_rel3.tsv
 ```
Or you can directly use the preprocessed [data](https://github.com/nusnlp/nea/tree/master/data) to run the code (this is what I did actually).

### Options

You can see the list of available options by running:

```bash
python main_nea.py -h
```

### Example

Here gives a command that could achieve the same effect as the example command in the ***Example*** part of the original repo:

>  The following command trains a model for prompt 1 in the ASAP dataset, using the training and development data from fold 0 and evaluates it.

```bash
python main_nea.py
    -dp ./data/
    -f 0 -p 1
    -o p1f0_result
    --emb embeddings.w2v.txt
    --type regp
    --cnndim 50
    --rnndim 300
```
