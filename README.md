# gan_practice

画像生成の実行用のスクリプトです

元github

SPADE:https://github.com/NVlabs/SPADE

FastGAN:https://github.com/odegeasslbc/FastGAN-pytorch

元論文

SPADE:https://arxiv.org/abs/1903.07291

FastGAN:https://arxiv.org/abs/2101.04775 (light weight GAN)

## SPADE base、FastGAN base について

generatr の構造は
FastGANを元にSAPDEを組み合わせたもの

- SPADE base

SPADE のdiscriminator とloss関数を採用
- FastGAN base

FastGAN のdiscriminator とloss関数を採用


## 準備

```
git clone https://github.com/ryuji0398/gan_practice.git
cd gan_practice
pip install -r requirements.txt
```

SPADE_base の設定

SPADE_base の実行にはSynchronized-BatchNorm-PyTorchも必要 (SPADEの中身で使う)
```
cd SPADE_base/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../../
```

FastGAN_base の設定

FastGAN_base の実行にはSynchronized-BatchNorm-PyTorchも必要 (SPADEの中身で使う)
```
cd FastGAN_base/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../../
```

## dataset　について

### coco-stuff

現状,coco-stuffのdatasetを使って学習を行なっている

```
datasetの形
./datasets/coco2017
├── train_img
│   └── (cococ2017の画像)
├── train_inst
│   └── (SAPDEで生成するもの,入力になる？？)
├── train_label
│   └── (coco_stuffのlabel)
├── val_img
├── val_inst
├── val_label
└── annotations
```


datasetの保存pathへ移動
./datasets/coco2017/ にdatasetsを保存する場合はこちらで移動
```
cd datasets/coco2017
```

保存する場所以下でdataを取得

coco の train2017の画像を取得
```
wget http://images.cocodataset.org/zips/train2017.zip
# 解凍
unzip train2017.zip
# train2017 と保存されるので train_img とフォルダの名前を変える
mv train2017 train_img 
```

coco の val2017の画像を取得
```
wget http://images.cocodataset.org/zips/val2017.zip
# 解凍
unzip val2017.zip
# val2017 と保存されるので val_img とフォルダの名前を変える
mv val2017 val_img 

```

coco-stuff labelmap　の取得
```
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
# 解凍
unzip stuffthingmaps_trainval2017.zip
# train2017 と val2017 と保存される(たしか...) ので フォルダの名前をそれぞれ変える
mv train2017 train_label
mv val2017 val_label

```

coco のアノテーション の取得
```
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# 解凍
unzip annotations_trainval2017.zip

```


## 実行について

#### SPADE_base　実行コマンド,設定方法

```
train　コマンド
python train.py --name [experiment_name] --dataset_mode coco --dataroot [path_to_coco_dataset] 

test コマンド
python train.py --name [experiment_name] --dataset_mode coco --dataroot [path_to_coco_dataset]
```


#### FastGAN_base 実行コマンド,設定方法



## フォルダ
```
./
├── README.md
├── SPADE_base
├── FastGAN_base
│   └── (FastGAN_base)
├── datsets
│   ├── coco2017/
│   └── coco_generate_instance_map.py (coco data においてdataの準備のコード)
└── requirements.txt
```
