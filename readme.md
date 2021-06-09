## Introduction

yNet: A Breast Ultrasound Image Classification Algorithm Based on Metric Learning.

## Requirements

- python >= 3.9
- pytorch >= 1.7
- pytorch-lightning == 1.3.1
- opencv-python
- rich
- omegaconf

## Usage

### compile datasets

See scripts in `src/data/script` for detail.

Here is an example of compiling BUSI using vscode task:
~~~ json
{
    "label": "make dataset: BUSI", 
    "type": "shell",
    "command": [
        "python src/data/script/makelabel.py BUSI --sets benign:0 malignant:1 --title Ym mask;",
        "python src/data/script/resize.py BUSI --sets benign malignant;",
        "python src/data/script/makecaches.py BUSI --sets benign malignant --title Ym mask --stat Ym;",
    ],
    "problemMatcher": []
}
~~~

### training

We recommend to train yNet on a GPU with __18000M free memory__. Our implement uses RTX3090. 

Multi-processing is not implemented. So DDP isn't available. We warn of using DP directly, since triplet loss and BN are used. 
Fork is welcomed.

~~~ shell
python3.9 src/train_toynetv1.py # paths.version=QAQ
~~~

You may need to edit `.env` on Linux.

@import "./data/README.md"
