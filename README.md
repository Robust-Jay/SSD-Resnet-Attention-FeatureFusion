# SSD Improvements

This repository is forked from the project [SSD](https://github.com/lufficc/SSD).
I improve SSD from the following aspects:
 1) add [Resnet](https://arxiv.org/abs/1512.03385) backbone (Resnet50 and Resnet101)
 2) add attention mechanism ([Squeeze-and-Excitation Module](https://arxiv.org/abs/1709.01507) , [Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521))
 3) add [feature fusion module](https://arxiv.org/abs/1712.00960v1)

## Train

### Setting Up Datasets
#### Pascal VOC

For Pascal VOC dataset, make the folder structure like this:
```
VOC_ROOT
|__ VOC2007
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ VOC2012
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ ...
```
Where `VOC_ROOT` default is `datasets` folder in current project, you can create symlinks to `datasets` or `export VOC_ROOT="/path/to/voc_root"`.

#### COCO

For COCO dataset, make the folder structure like this:
```
COCO_ROOT
|__ annotations
    |_ instances_valminusminival2014.json
    |_ instances_minival2014.json
    |_ instances_train2014.json
    |_ instances_val2014.json
    |_ ...
|__ train2014
    |_ <im-1-name>.jpg
    |_ ...
    |_ <im-N-name>.jpg
|__ val2014
    |_ <im-1-name>.jpg
    |_ ...
    |_ <im-N-name>.jpg
|__ ...
```
Where `COCO_ROOT` default is `datasets` folder in current project, you can create symlinks to `datasets` or `export COCO_ROOT="/path/to/coco_root"`.

### Train sample 1
    Configs:
        Backbone: Resnet50
        Input size: 300
        SE: False
        CBAM: False
        FUSION: False
    Run:
        python train.py --config-file configs/resnet50_ssd300_voc0712.yaml
        
### Train sample 2
    Configs:
        Backbone: Resnet50
        Input size: 300
        SE: False
        CBAM: False
        FUSION: True
    Run:
        python train.py --config-file configs/resnet50_ssd300_voc0712_feature_fusion.yaml
        
You can do custom training by modifying the .yaml files.
