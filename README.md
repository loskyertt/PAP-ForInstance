# Prototype Adaption and Projection for Few- and Zero-shot 3D Point Cloud Semantic Segmentation [[pdf](https://arxiv.org/pdf/2305.14335.pdf)]


## Overview

![framework](framework3d.png)



## Running

**Installation and data preparation please follow [attMPTI](https://github.com/Na-Z/attMPTI).**



### Training

Pretrain the segmentor which includes feature extractor module on the available training set:

```bash
bash scripts/pretrain_segmentor.sh
```

Train our method under few-shot setting:

```bash
bash scripts/train_PAP.sh
```

Train our method  under few-and zero-shot setting:

```bash
bash scripts/train_PAPFZ.sh
```

### Evaluation

Test our method  under zero-shot setting:

```bash
bash scripts/eval_PAPFZ.sh
```

Test our method under few-shot setting:

```bash
bash scripts/eval_PAP.sh
```

Note that the above scripts are used for 2-way 1-shot on S3DIS (S^0). Please modify the corresponding hyperparameters to conduct experiments on other settings.



## Citation
Please cite our paper if it is helpful to your research:

    @article{PAPFZS3D,
      title={Prototype Adaption and Projection for Few- and Zero-shot 3D Point Cloud Semantic Segmentation},
      author={He, Shuting and Jiang, Xudong and Jiang, Wei and Ding, Henghui},
      journal={IEEE Transactions on Image Processing},
      year={2023},
      publisher={IEEE}
    }


## Acknowledgement
We thank [DGCNN (pytorch)](https://github.com/WangYueFt/dgcnn/tree/master/pytorch) and [attMPTI](https://github.com/Na-Z/attMPTI) for sharing their source code.

---

# 补充

## 目录结构

```bash
.
├── LICENSE
├── README.md
├── dataloaders
│   ├── __init__.py
│   ├── get_embedding.py
│   ├── loader.py
│   ├── s3dis.py
│   └── scannet.py
├── datasets
│   ├── S3DIS
│   │   └── meta
│   │       └── s3dis_classnames.txt
│   └── ScanNet
│       └── meta
│           ├── scannet_classnames.txt
│           └── scannetv2-labels.combined.tsv
├── framework3d.png
├── main.py
├── models
│   ├── __init__.py
│   ├── attention.py
│   ├── dgcnn.py
│   ├── dgcnn_new.py
│   ├── gmmn.py
│   ├── mpti.py
│   ├── mpti_learner.py
│   ├── proto_learner.py
│   ├── proto_learner_FZ.py
│   ├── protonet.py
│   ├── protonet_FZ.py
│   └── protonet_QGPA.py
├── preprocess
│   ├── collect_s3dis_data.py
│   ├── collect_scannet_data.py
│   └── room2blocks.py
├── runs
│   ├── __init__.py
│   ├── eval.py
│   ├── fine_tune.py
│   ├── mpti_train.py
│   ├── pre_train.py
│   └── proto_train.py
├── scripts
│   ├── eval_PAP.sh
│   ├── eval_PAPFZ.sh
│   ├── pretrain_segmentor.sh
│   ├── train_PAP.sh
│   └── train_PAPFZ.sh
└── utils
    ├── __init__.py
    ├── checkpoint_util.py
    ├── cuda_util.py
    └── logger.py
```
