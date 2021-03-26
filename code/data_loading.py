import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from utils import center_crop_square, square_pad_image, keep_one_dim_square
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import apply_affine_transform


class ImageSequence(Sequence):
    valid_file_ext = ['jpg', 'jpeg', 'png']

    def __init__(self,
                 dataset_folder,
                 batch_size,
                 target_split,
                 ims_size=224,
                 square_ims='center_crop',
                 only_brad=False,
                 seed=18,
                 augment=False):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.batch_size = batch_size
        self.dataset_folder = dataset_folder
        self.ims_size = ims_size
        self.only_brad = only_brad
        self.x_tr, self.x_val, self.x_test = self.generate_splits()
        self.target_split = target_split
        self.splits = {'train': self.x_tr, 'val': self.x_val, 'test': self.x_test}
        self.augment = augment
        if target_split == 'train':
            self.x = np.array([x[1] for x in self.x_tr])
            self.y = np.array([x[0] for x in self.x_tr])
        elif target_split == 'val' or target_split == 'validation' or target_split == 'valid':
            self.x = np.array([x[1] for x in self.x_val])
            self.y = np.array([x[0] for x in self.x_val])
        elif target_split == 'test':
            self.x = np.array([x[1] for x in self.x_test])
            self.y = np.array([x[0] for x in self.x_test])
        else:
            raise Exception('invalid split')

        self.image_loading_fn = self.get_image_loading_function(square_ims)
        self.class_id_mapper = self.generate_classes_and_inds()
        self.class_desc_mapper = self.generate_classes_and_desc()
        self.on_epoch_end()

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
        return list(self.generate_classes_and_inds(folder=True).values())

    def generate_classes_and_inds(self, folder=False):
        json_data = pd.read_json('{}/{}'.format(self.dataset_folder, 'cat_ids_desc.json'))
        mapper = {}
        for cat, idd in zip(json_data['cat'].values, json_data['id'].values):
            if folder:
                mapper[cat] = '{}/{}'.format(self.dataset_folder, idd)
            else:
                mapper[cat] = '{}'.format(idd)
        return mapper

    def generate_classes_and_desc(self):
        json_data = pd.read_json('{}/{}'.format(self.dataset_folder, 'cat_ids_desc.json'))
        mapper = {}
        for cat, idd in zip(json_data['cat'].values, json_data['desc'].values):
            mapper[cat] = '{}'.format(idd)
        return mapper

    def get_desc_for_class(self, c):
        return self.class_desc_mapper[c]

    def get_id_for_class(self, c):
        return self.class_id_mapper[c]

    def n_classes(self):
        c_and_folder = self.generate_classes_and_inds()
        return len(c_and_folder)

    def generate_splits(self, train_size=0.75, val_size=0.2, test_size=0.5):
        print('generating splits')
        c_and_folder = self.generate_classes_and_inds(folder=True)
        train_ims_and_classes = []
        val_ims_and_classes = []
        test_ims_and_classes = []
        if not self.only_brad:
            for c, main_folder in tqdm(c_and_folder.items()):
                valid_ims = sorted(self.list_valid_files(main_folder))
                n_ims = len(valid_ims)
                cut_tr = math.floor(n_ims * train_size)
                cut_val = math.floor(n_ims * (train_size + val_size))
                train_ims_and_classes += [[c, vi] for vi in valid_ims[:cut_tr]]
                val_ims_and_classes += [[c, vi] for vi in valid_ims[cut_tr:cut_val]]
                test_ims_and_classes += [[c, vi] for vi in valid_ims[cut_val:]]
            return train_ims_and_classes, val_ims_and_classes, test_ims_and_classes
        else:
            c = [0]
            print('only brad!')
            for main_folder in tqdm([v for k, v in c_and_folder.items() if k != 1000]):
                valid_ims = sorted(self.list_valid_files(main_folder))
                r_ims = np.random.choice(valid_ims, 8)
                train_ims_and_classes += [[c, vi] for vi in r_ims[:3]]
                val_ims_and_classes += [[c, vi] for vi in valid_ims[3:6]]
                test_ims_and_classes += [[c, vi] for vi in valid_ims[6:8]]
            c = [1]
            main_folder = c_and_folder[1000]
            valid_ims = sorted(self.list_valid_files(main_folder))
            print('total 1 ims: {}'.format(len(valid_ims)))
            n_ims = len(valid_ims)
            cut_tr = math.floor(n_ims * train_size)
            cut_val = math.floor(n_ims * (train_size + val_size))
            train_ims_and_classes += [[c, vi] for vi in valid_ims[:cut_tr]]
            val_ims_and_classes += [[c, vi] for vi in valid_ims[cut_tr:cut_val]]
            test_ims_and_classes += [[c, vi] for vi in valid_ims[cut_val:]]
            return train_ims_and_classes, val_ims_and_classes, test_ims_and_classes

    def split_sizes(self):
        return len(self.x_tr), len(self.x_val), len(self.x_test)

    def get_random_sample(self, target=None, cat=None):
        if target:
            c = np.random.choice(np.arange(len(self.splits[target])))
            if cat:
                ii = np.random.choice([i for i, x in enumerate(self.splits[target]) if x[0] == cat])
                return self.splits[target][ii]
            return self.splits[target][c]
        else:
            c = np.random.choice(range(len(self.x)))
            return [self.y[c], self.x[c]]

    def load_im_and_proc(self, im_path):
        batch_x = []
        for filename in [im_path]:
            pil_im = self.image_loading_fn(filename)
            im_ar = image.img_to_array(pil_im)
            batch_x.append(im_ar)
        batch_x_proc = preprocess_input(np.array(batch_x))
        return batch_x_proc

    def custom_aug_params(self):
        zx, zy = np.random.uniform(1 - 0.1, 1 + 0.1, 2)
        shear = np.random.uniform(-10, 20)
        theta = np.random.uniform(-20, 20)
        transform_parameters = {'theta': theta,
                                'shear': shear,
                                'zx': zx,
                                'zy': zy}
        return transform_parameters

    def on_epoch_end(self):
        permutation = np.random.permutation(np.arange(len(self.x)))
        self.x = self.x[permutation]
        self.y = self.y[permutation]

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x_files = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = []
        for filename in batch_x_files:
            # TODO: add some more augmentation here!
            if self.augment:
                loading_fn = np.random.choice([self.image_loading_fn, keep_one_dim_square])
            else:
                loading_fn = self.image_loading_fn
            pil_im = loading_fn(filename)
            im_ar = image.img_to_array(pil_im)
            if self.augment:
                aug_params = self.custom_aug_params()
                im_ar = apply_affine_transform(im_ar, **aug_params)
            batch_x.append(im_ar)
        batch_x_proc = preprocess_input(np.array(batch_x))
        return batch_x_proc, batch_y
