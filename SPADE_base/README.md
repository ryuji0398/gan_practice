
# SPADE_base

画像生成の実行用のコードでSPADEをベースに作られています。

generatr の構造は FastGANを元にSAPDEを組み合わせたもの

SPADE のdiscriminator とloss関数を採用

### 実行コマンド

基本の実行コマンド
- coco dataset を使用する場合

```
# 学習コマンド
python train.py --name [name] --dataset_mode coco --dataroot [data_path] --batchSize [バッチサイズの指定] 

# テストコマンド
python test.py --name [name] --dataset_mode coco --dataroot [data_path] 

```

dataroot 以下のフォルダは以下のようにしておく必要がある

```
dataroot/
|
|-- train_img   : 実画像
|-- train_label : label 画像（coco stuffのもの）
|-- train_inst  : instance 前処理で作成したもの
|-- val_img
|-- val_label
|-- val_inst

```


テストコマンドについて
以下のもので追加に指定することも可能、他にもある
```
--results_dir : 結果の出力先
--how_many    : 出力する画像枚数（生成する枚数）
--which_epoch : いつのモデルを使うか（指定がなければdefaltで最後のものが使われる）

```

- custom dataset を使用する場合

```
# 学習コマンド
python train.py --name [name] --dataset_mode custom --label_dir [label_path] --image_dir [img_path] --label_nc [ラベルの種類] --batchSize [バッチサイズの指定] --no_instance

```

custom data での実行に関しては現状 2種類のみのクラスでしかおそらく実行できない（あるクラスとその他）

label の入力については 2種類のもののみしか考えていないので　0 もしくは 255 の値が入っているグレースケールの画像にする必要がある



### フォルダ説明

```

./
|
|-- checkpoints 
|-- data        
|-- datasets    
|-- docs
|-- models
|-- options
|-- trainers
|-- utils
|
|-- test.py
└── train.py
```
