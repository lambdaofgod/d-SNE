import os
import argparse
import numpy as np
import struct
import pickle as pk
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import requests


class DatasetGenerator:
    def __init__(self):
        pass

    def download_file(self, url: str, storage_dir: Path, chunk_size=1024):
        file_name = url.split('/')[-1]
        file_path = storage_dir / file_name
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


class MNIST(DatasetGenerator):
    def __init__(self, root_dir: Path):
        super().__init__()
        self.root_dir = root_dir

    def generate(self):
        self.download()
        self.read()
        self.dump()

    def download(self,
                 train_images_url: str = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                 train_labels_url: str = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                 test_images_url: str = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                 test_labels_url: str = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
                 ):
        for url in [train_images_url, train_labels_url, test_images_url, test_labels_url]:
            file_path = self.download_file(url, self.root_dir)
            self.unzip_file(file_path)

    def read(self):
        def read_img(file_path: Path):
            with open(str(file_path), 'rb') as fimg:
                magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
                img = np.fromfile(fimg, dtype=np.uint8).reshape(-1, rows, cols)

            return img

        def read_lbl(file_path: Path):
            with open(str(file_path), 'rb') as flbl:
                magic, num = struct.unpack(">II", flbl.read(8))
                lbl = np.fromfile(flbl, dtype=np.int8)

            return lbl

        train_img = read_img(self.root_dir / 'train-images-idx3-ubyte')
        train_lbl = read_lbl(self.root_dir / 'train-labels-idx1-ubyte')

        test_img = read_img(self.root_dir / 't10k-images-idx3-ubyte')
        test_lbl = read_lbl(self.root_dir / 't10k-labels-idx1-ubyte')

        self.dataset = {'TR': [train_img, train_lbl], 'TE': [test_img, test_lbl]}

    def dump(self):
        with open(os.path.join(self.root_dir, self.__class__.__name__ + '.pkl'), 'wb') as fout:
            pk.dump(self.dataset, fout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('-d', '--dataset', default='mnist', help='dataset')

    args = parser.parse_args()

    root_dir = Path(args.dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset.lower() == 'mnist':
        dataset = MNIST(root_dir)
        dataset.generate()
    else:
        print('Required download the dataset and packed by yourself, sorry for inconvenience')


if __name__ == '__main__':
    main()
