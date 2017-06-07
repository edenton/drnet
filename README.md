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


##  Training on KTH
First download the KTH action recognition dataset by running:
```
sh datasets/download_kth.sh /my/kth/data/path/
```
where /my/kth/data/path/ is the directory the data will be downloaded into. Next, convert the downloaded .avi files into .png's for the data loader. To do this you'll want [ffmpeg](https://ffmpeg.org/) installed. Then run:
```
th convert_kth --dataRoot /my/kth/data/path/ --imageSize 128
```
The ```--imageSize``` flag specifiec the image resolution. Experimental results in the paper used 128x128, but you can also train a model on 64x64 and it will train much faster.
Now you're ready to train the DrNet model by running:
```
th train_drnet_skip --dataRoot /my/kth/data/path/ --imageSize 128 --nThreads 2
``` 
Setting ```--nThreads``` utilizes multithreaded data loading and will speed up training significantly.
