import os
import pickle

import numpy as np
import torch

import lib

##   Assumes:
##     num_nan_policy=='mean'
##     cat_nan_policy=='new'
##     cat_policy=='indices'
class Normalization:
    def __init__(self,args,seed):
        print('Loading normalization data...')
        dataset_dir = lib.get_path(args['data']['path'])
        self.dataset_info = lib.load_json(dataset_dir / 'info.json')

        normalization=args['data'].get('normalization')
        num_nan_policy='mean'
        cat_nan_policy='new'
        cat_policy=args['data'].get('cat_policy', 'indices')
        cat_min_frequency=args['data'].get('cat_min_frequency', 0.0)
        normalizer_path = dataset_dir / f'normalizer_X__{normalization}__{num_nan_policy}__{cat_nan_policy}__{cat_policy}__{seed}.pickle'
        encoder_path = dataset_dir / f'encoder_X__{normalization}__{num_nan_policy}__{cat_nan_policy}__{cat_policy}__{seed}.pickle'
        if cat_min_frequency:
            normalizer_path = normalizer_path.with_name(
                normalizer_path.name.replace('.pickle', f'__{cat_min_frequency}.pickle')
            )
            encoder_path = encoder_path.with_name(
                encoder_path.name.replace('.pickle', f'__{cat_min_frequency}.pickle')
            )
            ### some of these files may not exist; e.g. num_new_values exists only if the training DS contains NAs
        if os.path.exists(dataset_dir / f'num_new_values.npy'):
            self.num_new_values = np.load(dataset_dir / f'num_new_values.npy')
        else:
            self.num_new_values = np.zeros(self.dataset_info['n_num_features'])
        self.normalizer = pickle.load(open(normalizer_path, 'rb'))
        self.encoder = pickle.load(open(encoder_path, 'rb'))
        self.max_values = np.load(dataset_dir / f'max_values.npy')
        ## y_mean, y_std
        self.y_mean_std = np.load(dataset_dir / f'y_mean_std.npy')
        self.cat_values = np.load(dataset_dir / f'categories.npy').tolist()

        self.n_num_features = self.dataset_info['n_num_features']
        self.n_cat_features = self.dataset_info['n_cat_features']

    def normalize_x(self,x_num,x_cat):
        ## (4.1) transform numerical data
        ## (4.1.1) replace nan by mean
        num_nan_mask = np.isnan(x_num)
        if num_nan_mask.any():
            num_nan_indices = np.where(num_nan_mask)
            x_num[num_nan_indices] = np.take(self.num_new_values, num_nan_indices[1])

        ## (4.1.2) normalize
        x_num = self.normalizer.transform(x_num)
        x_num = torch.as_tensor(x_num, dtype=torch.float)

        ## (4.2) transform categorical data
        ## (4.2.1) replace nan
        x_cat = np.array(x_cat,dtype='<U32').reshape(1,-1)
        cat_nan_mask = x_cat == 'nan'
        if cat_nan_mask.any():
            cat_nan_indices = np.where(cat_nan_mask)
            x_cat[cat_nan_indices] = '___null___'

        ## (4.2.2) encode; fix values, since new data may be out of cat range
        unknown_value = self.encoder.get_params()['unknown_value']
        x_cat = self.encoder.transform(x_cat)

        ## this won't work, since the transformer can't handle max_values[column_idx]+1; make sure that train contains nan's if they're present
        for column_idx in range(x_cat.shape[1]):
            x_cat[x_cat[:,column_idx]==unknown_value,column_idx] = ( self.max_values[column_idx]+1 )
        x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        return x_num, x_cat

    def normalize_y(self,y_raw):
        return (y_raw*self.y_mean_std[1])+self.y_mean_std[0]

    def get_example_data(self):
        if self.n_num_features == 8:
            print("Assuming ACOTSP dataset.")
            x_num=[1.83, 7.87, 0.69, 36.0, np.nan, np.nan, np.nan, 6.0 ]
            x_cat=['129', 'as', '2', 'nan' ]
        else:
            print("Assuming LKH dataset.")
            x_num=[255.0, 0.0, 5.0, 5.0, 4.0, 3.0, 12.0, 14.0, 20.0, 5.0, 986.0, 5.0]
            x_cat=['121', 'NO', 'QUADRANT', 'QUADRANT', 'YES', 'YES', 'GREEDY', 'NO', 'NO', 'YES']
        x_num=np.array(x_num).reshape(1,-1)
        return x_num,x_cat
