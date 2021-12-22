## [Weakly supervised discriminative feature learning with state information for person identification](https://arxiv.org/pdf/2002.11939.pdf)

<img src="img/intro.png" width="400"/>

This repo contains the source code for our CVPR'20 work
**Weakly supervised discriminative feature learning with state information for person identification** 
([paper](https://arxiv.org/pdf/2002.11939.pdf).
Our implementation is based on Pytorch.
In the following is an instruction to use the code
to train and evaluate our model.

![](img/framework.png)

### Prerequisites
1. [Pytorch 1.0.0](https://pytorch.org/)
2. Python 3.6+
3. Python packages: numpy, scipy, pyyaml/yaml, h5py

### Data and pretrained weights
Please download the Market/Duke/Multi-PIE/CFP dataset
as well as the pretrained ResNet50 weights from
[BaiduPan](https://pan.baidu.com/s/1O0s_dJcbkku6T0MwlLQecw) with
password "tih8". 
Put all of them (datasets and pretrained weights) into */data/* (please create a folder /data in the root directory).

### Run the code

Please enter the main folder, and run
```bash
python src/main.py --gpu 0,1,2,3 --save_path runs/market
```
on Market dataset,
where "0,1,2,3" specifies your gpu IDs.
If you are using gpus with 12G memory, you need 4 gpus to run 
in the default setting (batchsize=384).
Note that **small batch size is NOT recommended** as it increases the variance in estimating in-batch feature distributions.
If you have to set a small batch size, please lower the learning rate as the gradient
would be stronger for a smaller batch size.
Please also note that since I load the whole datasets into cpu memory and parallelize computation,
you need at least 12G RAM memory for Market. Hence I recommend you run it on a server.

For Duke dataset, run
```bash
python src/main.py --gpu 0,1,2,3 --save_path runs/duke
```
For Multi-PIE:
```bash
python src/main_mpie.py --gpu 0,1,2,3 --save_path runs/mpie
```
For CFP:
```bash
python src/main_cfp.py --gpu 0,1,2,3 --save_path runs/cfp
```

### Main results
We find our method can achieve performances that are comparable to standard supervised fine-tuning performances on Duke, MultiPIE and CFP datasets.
#### Duke
Method |Rank-1|Rank-5|MAP
-|-|-|-
Supervised fine-tune| 75.0|85.0|57.2
Pretrained| 43.1| 59.2| 28.8
Ours| 72.1|83.5| 53.8
#### Market
Method |Rank-1|Rank-5|MAP
-|-|-|-
Supervised fine-tune| 85.9|95.2|66.8
Pretrained| 46.2| 64.4| 24.6
Ours| 74.0|87.4| 47.9
#### Multi-PIE
Method| avg| 0&deg;| 15&deg;| 30&deg;| 45&deg;| 60&deg;
-|-|-|-|-|-|-
Supervised fine-tune| 98.2| 99.7|99.4|98.8|98.1|95.7
Pretrained| 88.7| 98.5| 97.5| 93.7| 89.7| 71.2
Ours| 97.1| 99.1| 98.9| 98.3| 96.8| 93.1
#### CFP
Method| Accuracy| EER (lower better)| AUC
-|-|-|-
Supervised fine-tune| 95.5| 4.7| 98.8
Pretrained| 92.9| 7.4| 97.8
Ours| 95.5| 4.7| 98.8

### Trained models
Trained models can be found [here](https://mega.nz/#F!eI90mQaR!zol1E4Q5OX7i0yFLEVvNLQ). Note that I just trained these models
so they are slightly different (maybe higher/lower within 1% than reported numbers) 
from the models used to report results in the paper.
If you used the code and found the obtained numbers a bit different from the paper,
it is expected because the performance of unsupervised deep learning can fluctuate sometimes.

To evaluate the trained models, simply modify the pretrain_path in runs/dataset/args.yaml,
as well as setting epoch to 0.

### Reference

If you find our work helpful in your research,
please kindly cite our paper:

Hong-Xing Yu and Wei-Shi Zheng, "Weakly supervised discriminative feature learning with state information for person identification",
In CVPR, 2020.

bib:
```
@inproceedings{yu2020weakly,
  title={Weakly supervised discriminative feature learning with state information for person identification},
  author={Yu, Hong-Xing and Zheng, Wei-Shi},
  year={2020},
  booktitle={IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

### Contact
If you find any problem or question please kindly let me know by opening an issue or emailing me at xKoven@gmail.com 
