# DrNet 
### [[paper]](https://arxiv.org/abs/1705.10915) [[project page]](https://sites.google.com/view/drnet-paper//)   

Torch implementation for Unsupervised Learning of Disentangled Representations from Video.



# Training 
Currently only MNIST data loader implemented, more to come.

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



