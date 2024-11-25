
import os

def get_dataset_filepaths():
    directory = "./datasets"
    csv_filenames = []

    for file_name in os.listdir(directory):
        if not file_name.endswith(".csv"):
            continue

        csv_filenames.append(os.path.join(directory, file_name))

    return csv_filenames 

def get_linguistic_property(filename):
    return filename.split("_")[0]


def get_results_filepath(dataset_csv, metric):
    return os.path.join("results", get_linguistic_property(os.path.basename(dataset_csv)))