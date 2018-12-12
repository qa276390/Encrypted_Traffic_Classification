# Encrypted_Traffic_Classification

## Dependency
We use [joy 2.0](https://github.com/cisco/joy) tool to convert pcap file to json.


This project is dependent on Python 

- keras == 2.2.0
- numpy == 1.14.0
- pandas == 0.22.0
- matplotlib == 2.1.2
- scikit-learn == 0.19.1
- xgboost == 0.80

## Usage

### Folder and Dataset

To start this project you can download dataset from [VPN-nonVPN dataset (ISCXVPN2016)](https://www.unb.ca/cic/datasets/vpn.html)
and unzip those file to `./data/PCAP`.
 
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

You can convert pcap file to json, do:


```shell
python3 ??
```

Or using the json file we have already converted:

```shell
# To use the table generator you should decide 
# 3 paraneters : input folder, output table, malicious or not
sh multi_gen.sh ../data/JSON Table.csv 0
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
