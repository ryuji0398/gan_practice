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


## 実行について




## フォルダ
```
./
├── README.md
├── SPADE_base
├── FastGAN_base
│   └── FastGAN_base
└── requirements.txt
```
