import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from data_loading import ImageSequence
from utils import square_pad_image

from PIL import UnidentifiedImageError
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import warnings

warnings.filterwarnings('error')

def correct_sample(sample):
    sample['status_code'] = 'er'
    sample['fname'] = None
    return sample


def curate(folder_name, file_name='/curated_results.json'):
    ims_df = pd.read_json(folder_name + file_name)
    invalids = []
    ims = []
    for i, sample in ims_df[ims_df['status_code'] == 200].iterrows():
        try:
            if sample['fname'] is None:
                sample = correct_sample(sample)
                ims_df.iloc[i] = sample
            else:
                im = square_pad_image(sample['fname'])
                ims.append(np.asarray(im))
        except UnidentifiedImageError:
            sample = correct_sample(sample)
            ims_df.iloc[i] = sample
        except OSError:
            sample = correct_sample(sample)
            ims_df.iloc[i] = sample
        except Warning as w:
            print(w)
            print('Warning', sample['fname'])
            sample = correct_sample(sample)
            ims_df.iloc[i] = sample
        except Exception as e:
            print(e)
            print(sample['fname'])
            sample = correct_sample(sample)
            ims_df.iloc[i] = sample
            print('OTHER ERROR!')
            # raise Exception('BREAK other error')
    print(folder_name, len(ims))
    preprocess_input(np.array(ims))
    new_json_path = folder_name + '/curated_results.json'
    ims_df.to_json(new_json_path)


dataset_path = '/data/datasets/imagenet2012/'

train_generator = ImageSequence(dataset_path, 20, target_split='train', square_ims='center_pad')

folders = train_generator.list_valid_folders()

# single thread
# for folder_name in tqdm(folders):
#     curate(folder_name)

# parallel
from multiprocessing import Pool
pool = Pool(6)
list(tqdm(pool.imap(curate, folders[::-1]), total=len(folders)))
pool.close()
