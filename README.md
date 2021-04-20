# Microscopy Image Classification. 

This data is based on and obtained from the paper "[Accurate classification of protein subcellular localization from high throughput microscopy images using deep learning](https://kodu.ut.ee/~leopoldp/2016_DeepYeast/)". The specific dataset for this project can also be downloaded [here](https://kodu.ut.ee/~leopoldp/2016_DeepYeast/data/main.tar.gz). Additionally, the following files are also required to be downloaded:
* [HOwt_doc.txt](https://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/reports/HOwt_doc.txt)
* [HOwt_train.txt](https://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/reports/HOwt_train.txt)
* [HOwt_val.txt](https://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/reports/HOwt_val.txt)
* [HOwt_test.txt](https://kodu.ut.ee/~leopoldp/2016_DeepYeast/code/reports/HOwt_test.txt)

# Reproducing Our Work
## Reorganization steps
The downloaded data is originally very messy and some data reorganization is required. To easily reorganize the data, follow the steps below:

1. Clone the repository. This can be done using `git clone` in a folder you want to store this project as shown below.
```
$ git clone git@github.com:zengfung/MicroscopyImageClassification.git
```

2. Move the dataset and the HOwt files mentioned above into the MicroscopyImageClassification folder. The dataset has the name `main.tar.gz`.

3. Open up terminal, unzip the `main.tar.gz` file using the following command.
```
# Linux
$ tar xvzf main.tar.gz

# Mac OS
$ gunzip -c foo.tar.gz | tar xopf -

# Windows
$ tar xzvf main.tar.gz
```

4. Run `reorganize-data.py`. Wait approximately 5-10 minutes and the folders will be reorganized into the path `label/filename.png` while the original folders `plate/filename.png` will be deleted. Additionally, a CSV file `labels-files.csv` will be created to track the filenames, paths, labels, and label index.
```
$ python reorganize-data.py
```

**Note:** This should only be run once before the start of the entire project.