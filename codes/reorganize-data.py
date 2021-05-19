import os
import re
import shutil
import pandas as pd
import numpy as np

def get_filename_labels(imagedata_file, labels_file):
	"""
	Join the `imagedata_file` data frame with the `labels_file` data frame to 
	obtain a data frame with the following fields:
	- filename
	- label_idx
	- label
	"""
	image_df = pd.read_csv(
		imagedata_file, sep="\s+", header=None, names=["file", "label_idx"]
	)
	labels_df = pd.read_csv(labels_file, sep=";")
	df = image_df.set_index("label_idx").join(labels_df.set_index("label_idx"))
	df = df.reset_index()
	return df
	
def mkdir_by_class(df):
    """
	Based on the data frame `df`, sort image files into its respective folders,
	which are named by classes.
	"""
    nrows = df.shape[0]
    for i in range(nrows):
        label = df.loc[i, "label"]
        if not os.path.exists(label):
            os.makedirs(label)
        src = df.loc[i, "file"]
        dst = label + re.sub("^[a-z0-9]+", "", src)
        shutil.move(src,dst)
    return None

def rmdir_by_plate():
    """
	Remove folders plate01, plate02, etc as they're no longer needed.
	"""
    for i in range(1,12):
        foldername = "plate0"+str(i) if i<10 else "plate"+str(i)
        if os.path.exists(foldername):
            shutil.rmtree(foldername, ignore_errors=True)
    return None

def generate_csv():
    """
	Generates a csv file containing 4 fields:
	- label
	- label_idx
	- file
	- path
    - split
	"""
    filenames = []
    labels = []
    paths = []
    for _, dirs, _ in os.walk("."):
        for dir in dirs:
            for subroot, _, files in os.walk(dir):
                # obtain filename, labels, paths, and splits for each image
                for name in files:
                    filenames += [name]
                    labels += [dir]
                    paths += [subroot+"/"+name]
    # make data frame
    df = pd.DataFrame({"label": labels, "file": filenames, "path": paths})
    labels_df = pd.read_csv("HOwt_doc.txt", sep=";")
    # join with label_idx
    df = df.set_index("label").join(labels_df.set_index("label"))
    df = df.reset_index()
    # rearrange columns
    df = df[["label", "label_idx", "file", "path"]]
    return df

def main():
    """
	Reorganizes the image files from the folders plate01, plate02, etc into 
	folders named by the classes of the images they contain.
	"""
    # get directory of current file
    file_dir = os.path.dirname(os.path.realpath(__file__))
    # switch working directory to data directory 
    main_dir = os.path.dirname(file_dir)
    os.chdir(main_dir + "\\data")
    # reorganize data if it is not organized already
    labels_df = pd.read_csv("HOwt_doc.txt", sep=";")
    cur_folders = [x[1] for x in os.walk(os.getcwd())][0]
    exp_folders = ["plate0" + str(i) for i in range(1,10)] + ["plate10", "plate11"]
    sor_folders = labels_df.label.to_list()
    if cur_folders == exp_folders:      # data can be reorganized
        # collect filenames and labels
        dfs = []
        for filename in ["HOwt_test.txt", "HOwt_val.txt", "HOwt_train.txt"]:
            dfs.append(get_filename_labels(filename, "HOwt_doc.txt"))
        # insert the train/test/valid split column
        dfs[0]["split"] = "test"
        dfs[1]["split"] = "valid"
        dfs[2]["split"] = "train"
        # concatenate dfs to form one large df
        df = pd.concat(dfs, ignore_index=True)
        # make directories by classes to store images
        mkdir_by_class(df)
        # remove unneeded directories
        rmdir_by_plate()
        # generate csv to keep track of files and labels
        print("Generating labels-files.csv")
        labels_df = generate_csv()
        # join both data frames on file name to extract data split
        df["file"] = df["file"].str.split("/", expand=True)[1]
        df = df[["file", "split"]].set_index("file").join(labels_df.set_index("file"))
        df = df.reset_index()
        df = df[["split", "label", "label_idx", "file", "path"]]
        df.to_csv("labels-files.csv", index=False)
        print("Data reorganization completed!")
    elif cur_folders == sor_folders:    # data already reorganized
        print("Data already reorganized!")
    else:                               # data can't be reorganized
        print("Sorry, unexpected folders found!")
        raise Exception("Delete all folders in data/, unpack main.tar.gz and try again!")
    return None

if __name__ == '__main__':
	main()