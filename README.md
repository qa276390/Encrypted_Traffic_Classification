# Encrypted_Traffic_Classification

## Dependency
We use [joy 2.0](https://github.com/cisco/joy) tool to convert pcap file to json.

Also, we use [GNU parallel version 3](https://www.gnu.org/software/parallel/) to speed up the data preprocessing.


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
    |   └── ...                 
    ├── main
    │   ├── train.py              # main function to train
    │   ├── TRAIN.sh              # script to train
    |   └── ...                   
    └── ...

### Data Preprocessing 

(optional) You can convert pcap file to json, do:

Before execute the code below, please make sure there are ~/joy/bin/joy and ~/joy/sleuth files in your computer. The fourth input please fill in a integer between 0 to 200, which means the packet num in the flow. The example below extract the first 50 packets in a flow and transform the informations to json file.

```shell
# 4 parameters : input folder, temporary folder, output folder, maximun number of packets 
cd data && sh pcap_to_json.sh PCAP tmpJSON JSON 50
```

Or skipping above steps, using the json file we have already converted:

```shell
# To use the table generator you should decide 
# 3 parameters : input folder, output table, malicious or not
cd prepro &&　sh multi_gen.sh ../data/JSON Table.csv 0 && python3 toTrain.py Table.csv
```

### Training & Evaluation

To train a model, do:

```shell
# To train the model, you should specify the mode(DNN/XGB):
cd ./main && sh TRAIN.sh DNN
```

Or specify other parameters:

```shell
# There are up to 5 parameters: mode, data path, output folder, batch size, patience :
cd ./main && python3 train.py --mode DNN --source_data_folder ../data --output_folder ./output --batch_size 1024 --patience 1000 
```


