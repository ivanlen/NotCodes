import math
import numpy as np
import os
import pandas as pd

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from utils import center_crop_square, square_pad_image


class ImageSequence(Sequence):
    valid_file_ext = ['jpg', 'jpeg', 'png']

    def __init__(self, dataset_folder, batch_size, target_split, ims_size=224, square_ims='center_crop'):
        self.batch_size = batch_size
        self.dataset_folder = dataset_folder
        self.ims_size = ims_size
        x_tr, x_val, _ = self.generate_splits()
        if target_split == 'train':
            self.x = np.array([x[1] for x in x_tr])
            self.y = np.array([x[0] for x in x_tr])
        elif target_split == 'val' or target_split == 'validation' or target_split == 'valid':
            self.x = np.array([x[1] for x in x_val])
            self.y = np.array([x[0] for x in x_val])

        self.image_loading_fn = self.get_image_loading_function(square_ims)

    def get_image_loading_function(self, square_fn_name):
        if square_fn_name == 'center_crop':
            return center_crop_square
        elif square_fn_name == 'center_pad':
            return square_pad_image
        else:
            raise Exception('invalid loading function {}'.format(square_fn_name))

    def list_valid_files(self, folder):
        # return ['{}/{}'.format(folder, x) for x in os.listdir(folder) if x.split('.')[-1] in self.valid_file_ext]
        df = pd.read_json('{}/{}'.format(folder, 'curated_results.json'))
        valids = df[df['status_code'] == 200]['fname'].values
        return valids

    def list_valid_folders(self):
        sub_folders = ['{}/{}'.format(self.dataset_folder, x) for x in os.listdir(self.dataset_folder)]
        return sub_folders

    def generate_classes_and_inds(self):
        return {c: main_folder for c, main_folder in enumerate(sorted(self.list_valid_folders()))}

    def n_classes(self):
        c_and_folder = self.generate_classes_and_inds()
        return len(c_and_folder)

    def generate_splits(self, train_size=0.75, val_size=0.2, test_size=0.5):
        print('generating splits')
        c_and_folder = self.generate_classes_and_inds()
        train_ims_and_classes = []
        val_ims_and_classes = []
        test_ims_and_classes = []
        for c, main_folder in tqdm(c_and_folder.items()):
            valid_ims = sorted(self.list_valid_files(main_folder))
            n_ims = len(valid_ims)
            cut_tr = math.floor(n_ims * train_size)
            cut_val = math.floor(n_ims * (train_size + val_size))
            train_ims_and_classes += [[c, vi] for vi in valid_ims[:cut_tr]]
            val_ims_and_classes += [[c, vi] for vi in valid_ims[cut_tr:cut_val]]
            test_ims_and_classes += [[c, vi] for vi in valid_ims[cut_val:]]
        return train_ims_and_classes, val_ims_and_classes, test_ims_and_classes

    def split_sizes(self):
        tr_splits, val_splits, test_split = self.generate_splits()
        return len(tr_splits), len(val_splits), len(test_split)

    def preprocess_imgs_to_array(self):
        pass

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x_files = self.x[idx * self.batch_size:(idx + 1) *
                                                     self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        batch_x = []
        for filename in batch_x_files:
            pil_im = self.image_loading_fn(filename)
            im_ar = image.img_to_array(pil_im)
            batch_x.append(im_ar)
        batch_x_proc = preprocess_input(np.array(batch_x))
        return batch_x_proc, batch_y

    def on_epoch_end(self):
        permutation = np.random.permutation(np.arange(len(self.x)))
        self.x = self.x[permutation]
        self.y = self.y[permutation]
