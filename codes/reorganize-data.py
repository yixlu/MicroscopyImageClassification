import os
import re
import shutil
import pandas as pd

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
	"""
    filenames = []
    labels = []
    paths = []
    for root, dirs, _ in os.walk("."):
        for dir in dirs:
            for subroot, _, files in os.walk(dir):
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
    df.to_csv("labels-files.csv", index=False)
    return None

def main():
    """
	Reorganizes the image files from the folders plate01, plate02, etc into 
	folders named by the classes of the images they contain.
	"""
    # collect filenames and labels
    dfs = []
    for filename in ["HOwt_test.txt", "HOwt_val.txt", "HOwt_train.txt"]:
        dfs.append(get_filename_labels(filename, "HOwt_doc.txt"))
    df = pd.concat(dfs, ignore_index=True)
    # make directories by classes to store images
    mkdir_by_class(df)
    # remove unneeded directories
    rmdir_by_plate()
    # generate csv to keep track of files and labels
    generate_csv()
    return None

if __name__ == '__main__':
	main()