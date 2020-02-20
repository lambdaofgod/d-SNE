import os
import argparse
from typing import List

import numpy as np
import struct
import pickle as pk
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import requests


class DatasetGenerator:
    def __init__(self, root_dir: Path, name: str):
        self.root_dir = root_dir
        self.name = name

    def download_file(self, url: str, chunk_size=1024):
        file_name = url.split('/')[-1]
        file_path = self.root_dir / file_name
        with open(str(file_path), 'wb') as f:
            resp = requests.get(url, stream=True)
            file_size = resp.headers.get('content-length')
            print('Downloading from {}. Total bytes {}'.format(url, file_size))

            if file_size is None:
                f.write(resp.content)
            else:
                file_size = int(file_size)
                pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name)
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(chunk_size)
                pbar.close()
        return file_path

    def unzip_file(self, file_path: Path, clean_up: bool = True):
        file_path = str(file_path)
        with gzip.open(file_path, 'rb') as compressed_file:
            with open(file_path.replace('.gz', ''), 'wb') as uncompressed_file:
                shutil.copyfileobj(compressed_file, uncompressed_file)
        if clean_up:
            os.remove(file_path)

    def dump(self, train_img: np.ndarray, train_lbl: np.ndarray, test_img: np.ndarray, test_lbl: np.ndarray):
        dataset = [train_img, train_lbl, test_img, test_lbl]
        file_path = self.root_dir / (self.name + '.pkl')
        print('Creating pickle file {}...'.format(file_path))
        with open(str(file_path), 'wb') as f:
            pk.dump(dataset, f)

    def download_and_unzip(self, urls: List[str]):
        for url in urls:
            file_path = self.download_file(url)
            self.unzip_file(file_path)


def generate_mnist(root_dir: Path):
    ds_gen = DatasetGenerator(root_dir, 'mnist')

    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ]
    ds_gen.download_and_unzip(urls)

    def read_img(file_path: Path):
        with open(str(file_path), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(-1, rows, cols, 1)

        return img

    def read_lbl(file_path: Path):
        with open(str(file_path), 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

        return lbl

    train_img = read_img(root_dir / 'train-images-idx3-ubyte')
    train_lbl = read_lbl(root_dir / 'train-labels-idx1-ubyte')
    test_img = read_img(root_dir / 't10k-images-idx3-ubyte')
    test_lbl = read_lbl(root_dir / 't10k-labels-idx1-ubyte')

    ds_gen.dump(train_img=train_img, train_lbl=train_lbl, test_img=test_img, test_lbl=test_lbl)


def generate_usps(root_dir: Path):
    ds_gen = DatasetGenerator(root_dir, 'usps')

    urls = [
        'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz',
        'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.test.gz'
    ]
    ds_gen.download_and_unzip(urls)

    def read_images_and_labels(file_path: Path):
        with open(str(file_path), 'r') as f:
            data_array = np.loadtxt(f)
            labels = data_array[:, 0].astype(np.int)
            images = data_array[:, 1:].reshape(-1, 16, 16, 1)
        return images, labels

    train_img, train_lbl = read_images_and_labels(root_dir / 'zip.train')
    test_img, test_lbl = read_images_and_labels(root_dir / 'zip.test')

    ds_gen.dump(train_img=train_img, train_lbl=train_lbl, test_img=test_img, test_lbl=test_lbl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('-d', '--dataset')

    args = parser.parse_args()

    dataset_name = args.dataset.lower()

    root_dir = Path(args.dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    generators = {
        'mnist': generate_mnist,
        'usps': generate_usps
    }

    if dataset_name in generators:
        generators[dataset_name](root_dir)
        print('Done!')
    else:
        print('Dataset {} is not supported.'.format(dataset_name))


if __name__ == '__main__':
    main()
