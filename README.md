# Classification of subcellular protein locations in microscopic images using random forest machine learning: comparison of different feature extraction methods

Group members: Zeng Fung Liew, Yixing Lu

## Abstract
High-throughput microscopic images of single cells provide a rich source of biological data. However, given its high dimension nature, proper feature extraction methods need to be used to extract informative features from these images for downstream analysis. A common task in cell biology is to classify cells based on the subcelluar locations of target proteins using microscopic images. In the current project, images of single yeast cells with fluorescently-tagged endogenous protein were used to evaluate different feature extraction methods in terms of their performance in subcellular protein location classification, using a random forest model. Among the five techniques considered (scale-invariant feature transform, stationary wavelet transform, local discriminant basis, Haralick texture features, and scattering transform), scattering transform features achieved the highest test set classification accuracy: 71\%. Factors that restricted the performance of other feature extractors include low signal contrast in images, rotation and shift of cells, etc.

## Table of Contents
1. [Setup](#setup)
2. [Feature extraction codes](codes/)
3. [Data analysis and model fitting notebook](notebook/Methodology.ipynb)
4. [Report](report.pdf)

## Setup <a name="setup"></a>
The setup for reproducing our work follows 3 main parts: 
* Julia installation (our project imports Julia code via PyJulia)
* Dataset organization
* Activating our given environment using `environment.yml`

### Part 1: Installing Julia
This step is relatively straightforward. One can download Julia from their [official page](https://julialang.org/downloads/). At the time of writing, the stable version of Julia is v1.6.1.

The purpose of this step is to allow for the use of PyJulia later on. PyJulia is a library in Python that calls the codes from Julia. Once Julia is downloaded, you are set. There is no prior knowledge in Julia programming required as wrapper functions have been written in place for the purpose of this project.

### Part 2: Organizing dataset

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

### Part 3: Activating environment
Our project requires the use of various libraries. Depending on the versions used, some updated versions of certain libraries are not compatible with running this project, and therefore we recommend that one creates a conda environment that match ours.

*Note: Part 2 is assumed to be completed before continuing to Part 3.*

1. Navigate to the repo directory. You'll notice a file named `environment.yml`. Create the environment from the `environment.yml` file:
```shell
conda env create -f environment.yml
```

2. The environment name for this conda environment is `cells`. This environment has to be activated every time the project is run. To activate this environment, hit:
```shell
conda activate cells
```
3. You are done. You can verify that `cells` is activated using:
```shell
conda env list
```
