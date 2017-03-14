from os.path import isdir, isfile
from tqdm import tqdm
from urllib.request import urlretrieve
from zipfile import ZipFile

def download_text8_data():
    dataset_filename = 'data\\text8'
    dataset_folder_path = 'data'
    dataset_name = 'Text8 Dataset'
    dataset_zip = 'text8.zip'

    download_link = 'http://mattmahoney.net/dc/text8.zip'

    if not isfile(dataset_filename):
        urlretrieve(download_link, dataset_zip)

    if not isdir(dataset_folder_path):
        with ZipFile(dataset_zip) as zip_ref:
            zip_ref.extractall(dataset_folder_path)

    with open('data/text8') as f:
        text = f.read()

download_text8_data()
