### Pytorch 1.2 implementation of paper "TPU: Sparse and Non-uniform Point Cloud Upsampling with Transformer"

##  environment config
```bash
git clone https://github.com/TMingZhao/TPU_pytorch
cd TPU_pytorch

# conda environment
conda env create -f environment.yml
conda activate TPU_env

# compile
cd emd_pytorch_implement
python setup.py install
cd ..

```

## Train data
We use h5 file privided by [3PU](https://github.com/yifita/3PU_pytorch) for training, download it [here](https://polybox.ethz.ch/index.php/s/o3wzIrKlsngtwsb).

```bash
# train code
python train.py 

# run demo
python demo.py
```

## acknowledgement ##
This code based is created cortesy of [facebookresearch](https://github.com/facebookresearch/detr), [yanx27](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [yifita](https://github.com/yifita/3PU_pytorch), [liruihui](https://github.com/liruihui/PU-GAN)

