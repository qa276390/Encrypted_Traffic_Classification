# Encrypted_Traffic_Classification

## Dependency

This final project is dependent on Python 

- keras == 2.2.0
- numpy == 1.14.0
- pandas == 0.22.0
- matplotlib == 2.1.2
- scikit-learn == 0.19.1


## Usage

### Folder and Dataset

To start this project we have to download dataset from Kaggle and put those file to `./input/audio_train` and `./input/audio_test` .
 
    .
    ├── ...
    ├── data                    
    │   ├── PCAP                  # where pcap file should be
    │   ├── JSON                  # the output json file from joy
    │   └── ...
    ├── visualize 
    │   ├──                       # visualize
    ├── prepro
    │   ├── multi_gen.sh          # script to turn json file into table
    │   ├──                       # other prepro
    |   └── ...                 
    ├── main
    │   ├── train.csv             # training list
    │   ├── sample_submission.csv # testing list
    │   ├── audio_train           # Folder contains train data (wav file)
    │   ├── audio_test            # Folder contains test data (wav file)
    |   └── mfcc                  # output folder for mfcc_test.npy
    └── ...

### Data Preprocessing 

To use the data generator to append the dataset, do:

if only for predict:

```shell
python3 ./final/src/data_gen.py --test_only 1
```

for train and predict:

```shell
# To use the data generator you should decide 
# 2 paraneters : strech, num
python3 ./final/src/data_gen.py --strech 1.1 --num 5 
```

### Training

To train a model, do:

```shell
# To use the train you should give the model you want to use:
cd ./final/src && bash train.sh 1d_conv
cd ./final/src && bash train.sh 2d_mfcc
```


### Predict

```shell
cd ./final/src && bash predict.sh
```
whitch will output `1d_2d_ensembled_submission.csv` in the same directory





## Reference
