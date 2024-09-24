import os
import pandas as pd

def image_paths(path):
    '''
    Get class names, input file paths, and class labels from the dataset.

    Parameters:
    - path (str): Dataset folder path.

    Returns:
    - tuple: A tuple containing lists of class names, input file paths, and class labels.
    '''
    classes = []
    class_labels = []
    files = []
    folders = os.listdir(path)
    
    # Enumerate through all the folders, get file paths, and class labels
    for i, each_class in enumerate(folders):
        current_class_path = os.path.join(path, each_class)
        for each_file in os.listdir(current_class_path):
            classes.append(each_class)
            class_labels.append(i)
            files.append(os.path.join(current_class_path, each_file))
    return classes, files, class_labels

def convert_to_df(files, classes, class_labels):
    '''
    Convert filepaths, class labels to a DataFrame.

    Parameters:
    - files (list): List of file paths.
    - classes (list): Class labels (0 ... n).
    - class_labels (list): Class label name (type of cancer).

    Returns:
    - pd.DataFrame: DataFrame containing class names, file paths, and class labels.
    '''
    class_series = pd.Series(classes, name="class_name")
    class_label_series = pd.Series(class_labels, name="class")
    file_series = pd.Series(files, name="file_path")
    df = pd.concat([class_series, file_series, class_label_series], axis=1)
    return df

if __name__ == "__main__":

    classes, files, class_labels = image_paths('dataset')
    df = convert_to_df(files, classes, class_labels)
    df.to_csv("dataset.csv", index=False)
