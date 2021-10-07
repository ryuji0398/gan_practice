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


## 実行について

#### SPADE_base　実行コマンド,設定方法


#### FastGAN_base 実行コマンド,設定方法



## フォルダ
```
./
├── README.md
├── SPADE_base
├── FastGAN_base
│   └── FastGAN_base
└── requirements.txt
```
