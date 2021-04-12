## Introduction

yNet: A Breast Ultrasound Image Diagnosis Algorithm Based on Representation Learning with Small Samples.
> ~~A Computer Aided Denoisis Algorithom for Classifying Breast Ultrasound Images Based on Weakly-Supervised Few Shot Learning~~

@import "./dashboard.md"

## Requirements

- python 3.9
- pytorch 1.7+
- pytorch-lightning 1.2.6
- opencv
- rich
- omegaconf

## Usage

### compile datasets

~~~ shell
~~~

### training

~~~ shell
python3.9 ./src/train_toynetv1.py
~~~

## Data

> We've collected several datasets for BUS image. We list publicly available sets here. We use additional private datasets for training as well.

### Indirectly

- http://cvprip.cs.usu.edu/busbench/
  - email agreement required
- [dataset B](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)
  - email agreement required
- https://www.ultrasoundcases.info/cases/breast-and-axilla/benign-lesions/
  - no direct download provided

### Directly

- Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.
  - 133 normal, 437 malignant, 210 benign
  - [DOI](https://doi.org/10.1016/j.dib.2019.104863)
  - [direct link](https://scholar.cu.edu.eg/Dataset_BUSI.zip)

- Rodrigues, Paulo Sergio (2018), “Breast Ultrasound Image”, Mendeley Data, V1, doi: 10.17632/wmy84gzngw.1
  - 100 benign and 150 malignant
  - [DOI](https://doi.org/10.17632/wmy84gzngw.1)
  - [download page](https://data.mendeley.com/datasets/wmy84gzngw/1)

- ~~https://github.com/xbhlk/STU-Hospital/tree/master/Hospital~~
  - Imaging Department of the First Affiliated Hospital of Shantou University.
