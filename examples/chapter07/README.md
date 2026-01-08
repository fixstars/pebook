# 第07章のサンプルコード

## 実行方法

コマンドは `examples/chapter07` をカレントディレクトリと想定しています。

### データセットの用意

[mmdetectionのドキュメント](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/docs/en/advanced_guides/datasets/nuscenes.md)を参考にしてください。
このサンプルコードでは `./mmdetection3d/data` に配置することを想定しています。

### docker コンテナの build

```
git submodule update -i

echo UID=$(id -u) >>.env
echo GID=$(id -g) >>.env
echo USER=$USER >>.env
docker compose build
```

### docker コンテナ内で実行

#### 事前学習重みの作成

参考: <https://github.com/open-mmlab/mmdetection3d/tree/fe25f7a51d36e3702f961e198894580d83c4387b/projects/BEVFusion#demo>

```
wget https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth
mv swint-nuimages-pretrained.pth mmdetection3d/swint-nuimages-pretrained.pth
docker compose up -d
docker compose exec work /ch7/run.sh
docker compose down
```
