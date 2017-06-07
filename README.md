# DrNet 
### [[paper]](https://arxiv.org/abs/1705.10915) [[project page]](https://sites.google.com/view/drnet-paper//)   

Torch implementation for Unsupervised Learning of Disentangled Representations from Video.



# Training 
To train the base model run:
```
th train_drnet.lua 
```
or the model with skip connections between content encoder and decoder:
```
th train_drnet_skip.lua 
```


To train an LSTM on the pose vectors run:
```
th train_lstm.lua --modelPath /path/to/model/
```


# Dataset
## KTH
To download the KTH action recognition dataset run:
```
bash datasets/download_kth.lua /my/data/path/
```
and to split the .avi files into .png's for the data loader run
```
th convert_kth --dataRoot /my/data/path/
```
