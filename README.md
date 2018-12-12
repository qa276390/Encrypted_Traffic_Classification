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
- argparse == 3.2

## Usage

### Folder and Dataset

To start this project you can download dataset from [VPN-nonVPN dataset (ISCXVPN2016)](https://www.unb.ca/cic/datasets/vpn.html)
and unzip those file to `./data/PCAP`.
 
    .
    ├── ...
    ├── data                    
    │   ├── PCAP                  # where pcap file should be
    │   ├── JSON                  # the output json file from joy(after sleuth)
    |   ├── tmpJSON               # tmp json file from joy
    │   └── pcap_to_json.sh       # turn pcap to json file
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

Before execute the code below, please make sure there are ~/joy/bin/joy and ~/joy/sleuth files in your computer. The fourth input please fill in a integer between 0 to 200, which means the packet num in the flow. The example below extract the first 50 packets in a flow and transform the informations to json file.

```shell
sh pcap_to_json.sh PCAP tmpJSON JSON 50
```

Or using the json file we have already converted:

```shell
# To use the table generator you should decide 
# 3 paraneters : input folder, output table, malicious or not
cd prerpo &&　sh multi_gen.sh ../data/JSON Table.csv 0
```

### Training

To train a model, do:

```shell
# To use the train you should give the model you want to use:
cd ./main && python3 DNN.py --source_data_folder ../data --output_folder ./output
```


### Predict

```shell
cd ./final/src && bash predict.sh
```
whitch will output `1d_2d_ensembled_submission.csv` in the same directory





## Reference
