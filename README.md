# Efficient classification of microscopy images

Group members: Zeng Fung Liew, Yixing Lu

## Abstract
We plan to do a multi-class image classification data analysis project. The dataset is obtained from [http://www.cs.ut.ee/∼leopoldp/2016_DeepYeast](http://www.cs.ut.ee/∼leopoldp/2016_DeepYeast). It is a set of single cell images extracted by Pärnamaa et. al (2017) from the original image library described in Chong et. al (2015). Each sample is a 64*64 pixel RGB image which contains two channels: red channel (RFP) shows the cell outline and green channel (GFP) shows the different subcellular location of protein of interest. There are a total of 12 different subcellular protein location classes, with some class imbalance present. The classes are disjoint, meaning one image will be assigned to only one class. Single cell images will be used as training and test set for the project since cell segmentation is not the focus for the current project. The objective of the project is to classify the subcellular protein locations based on single cell fluorescent images.

The pipeline for our project goes as follows: preprocessing → model fitting → model ensembling. For preprocessing methods, we planned to work with feature detection and extraction (such as Scale-Invariant Feature Transform (SIFT), Wavelet Transforms, Wavelet Scattering Transforms, Speed-Up Robust Features etc) and some dimension reduction techniques (such as PCA, Kernel PCA, t-SNE, LDA). The Python libraries that we will mainly use will therefore be Scikit-Learn, OpenCV, PyWavelets, and possibly Tensorflow. The anticipated issue here is that the aforementioned pre-processing methods might not work well at all, and might result in subpar performances. To counter this issue, we plan to utilize stacking ensemble techniques on multiple subpar models to improve classification results. Ideally, we hope to obtain results that can rival the VGG network architecture discussed in the original paper, yet much more achievable on regular computers without GPUs, ie. less computationally expensive.

## Setup

1. Download the dataset and all its required files. Here are the links to download them:  
    * [Image data](https://kodu.ut.ee/~leopoldp/2016_DeepYeast/data/main.tar.gz)
    * [HOwt_doc.txt](https://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/reports/HOwt_doc.txt)
    * [HOwt_train.txt](https://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/reports/HOwt_train.txt)
    * [HOwt_val.txt](https://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/reports/HOwt_val.txt)
    * [HOwt_test.txt](https://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/reports/HOwt_test.txt)

2. The downloaded data is originally very messy and some data reorganization is required. To easily reorganize the data, first clone this repository. This can be done using `git clone` in a folder you want to store this project as shown below.

```shell
$ git clone git@github.com:zengfung/MicroscopyImageClassification.git
```

3. Move all the downloaded files from step 1 into the `data/` folder.

4. Open up terminal, navigate to the `data/` directory using the `cd` command and unzip the `main.tar.gz` file using the following command. Once this is done, you will see 11 folders named `plate01/` to `plate11/` in your `data/` directory.
```shell
# Linux
$ tar xvzf main.tar.gz

# Mac OS
$ gunzip -c foo.tar.gz | tar xopf -

# Windows
$ tar xzvf main.tar.gz
```

5. Run `codes/reorganize-data.py`. Wait approximately 5 minutes and the folders will be reorganized into the path `data/label/filename.png` while the original folders `data/plate/filename.png` will be deleted. Additionally, a CSV file `labels-files.csv` will be created to track the filenames, paths, labels, and label index.
```
$ python codes/reorganize-data.py
```
