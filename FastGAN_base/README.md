
# FastGAN_base

画像生成の実行用のコードでFastGANをベースに作られています。

generatr の構造は FastGANを元にSAPDEを組み合わせたもの

FastGAN のdiscriminator とloss関数を採用

### 実行コマンド

基本の実行コマンド

- coco dataset を使用する場合

```
# 学習コマンド
python train.py --name [name] --dataset_mode coco --dataroot [data_path] --batchSize [バッチサイズの指定] 


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


- custom dataset を使用する場合

```
# 学習コマンド
python train.py --name [name] --dataset_mode custom --label_dir [label_path] --image_dir [img_path] --label_nc [ラベルの種類] --batchSize [バッチサイズの指定] --no_instance

```

custom data での実行に関しては現状 2種類のみのクラスでしかおそらく実行できない（あるクラスとその他）

label の入力については 2種類のもののみしか考えていないので　0 もしくは 255 の値が入っているグレースケールの画像にする必要がある

学習用の引数はSPADEとの統合の兼ね合いで使えないものもある

学習のiteration指定は train.py の ```if __name__ == "__main__":　```以下の　args.iterで指定する必要がある

引数では指定することができない

<!-- 
テストコマンドについて
以下のもので追加に指定することも可能、他にもある
```
--results_dir : 結果の出力先
--how_many    : 出力する画像枚数（生成する枚数）
--which_epoch : いつのモデルを使うか（指定がなければdefaltで最後のものが使われる）

``` -->


### フォルダ説明

```

./
|
|-- benchmarking 
|-- docker     
|-- lpips 
|-- scripts
|-- SPADE         : SPADE の dataloaderとgenneratorの構造上必要であるため
|-- train_results : 実験結果が保存される
|-- diffaug.py
|-- eval.py
|-- models.py
|-- operation.py
└── train.py
```
