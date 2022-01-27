# %%
import sys

import pickle
import numpy as np
import zero

import lib
from ft_transformer import *


# %%
if __name__ == "__main__":
    ## predict a value
    ##   Assumes:
    ##     num_nan_policy=='mean'
    ##     cat_nan_policy=='new'
    ##     cat_policy=='indices'
    @torch.no_grad()
    def predict(x_num, x_cat, num_new_values, max_values, normalizer, encoder, y_mean_std, device):
        ## (4.1) transform numerical data
        ## (4.1.1) replace nan by mean
        num_nan_mask = np.isnan(x_num)
        if num_nan_mask.any():
            num_nan_indices = np.where(num_nan_mask)
            x_num[num_nan_indices] = np.take(num_new_values, num_nan_indices[1])

        ## (4.1.2) normalize
        x_num = normalizer.transform(x_num)
        x_num = torch.as_tensor(x_num)

        ## (4.2) transform categorical data
        ## (4.2.1) replace nan
        cat_nan_mask = [c == 'nan' for c in x_cat]
        print(cat_nan_mask)
        
        if cat_nan_mask.any():
            cat_nan_indices = np.where(cat_nan_mask)
            x_cat[cat_nan_indices] = '___null___'

        ## (4.2.2) encode; fix values, since new data may be out of cat range
        x_cat = encoder.transform(x_cat)
        for i in range(x_cat.shape[0]):
            if x_cat[i]==unknown_value:
                x_cat[i]=max_value[i]+1
        x_cat = torch.as_tensor(x_cat)

        ## (4.3) evaluate
        if device.type != 'cpu':
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
        y_raw = model(x_num,x_cat)

        ## (4.4) transform the output back
        return (y_raw*y_mean_std[1])+y_mean_std[0]

    ## (1) load the configuration, general setup
    args, output = lib.load_config()
    args['model'].setdefault('token_bias', True)
    args['model'].setdefault('kv_compression', None)
    args['model'].setdefault('kv_compression_sharing', None)

    seed=args['seed']
    zero.set_randomness(seed)

    ## (2) load the transformation data
    ## https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/
    ## somewhere in the original code: dump(normalizer, open('normalizer.pkl', 'wb'))
    print('Loading normalization data...')
    dataset_dir = lib.get_path(args['data']['path'])
    normalization=args['data'].get('normalization')
    num_nan_policy='mean'
    cat_nan_policy='new'
    cat_policy=args['data'].get('cat_policy', 'indices')
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0)
    ## tbd: modify paths for non-zero cat_min_frequency
    normalizer_path = dataset_dir / f'normalizer_X__{normalization}__{num_nan_policy}__{cat_nan_policy}__{cat_policy}__{seed}.pickle'
    encoder_path = dataset_dir / f'encoder_X__{normalization}__{num_nan_policy}__{cat_nan_policy}__{cat_policy}__{seed}.pickle'


    ### (TBD: some of these files may not exists; e.g. num_new_values exists only if the training DS contains nans)
    ## num_new_values
    num_new_values = np.load(dataset_dir / f'num_new_values.npy')
    normalizer = pickle.load(open(normalizer_path, 'rb'))
    encoder = pickle.load(open(encoder_path, 'rb'))
    ## max_values
    max_values = np.load(dataset_dir / f'max_values.npy')
    ## y_std, y_mean
    y_mean_std = np.load(dataset_dir / f'y_mean_std.npy')
    ## cat values
    cat_values = np.load(dataset_dir / f'categories.npy').tolist()
    
    ## (3) test data
    x_num=np.array([1.83, 7.87, 0.69, 36.0, np.nan, np.nan, np.nan, 6.0 ]).reshape(1, -1)
    x_cat=['129', 'as', '2', '1']

    ## (3) load the model (and possibly move it to the GPU)
    print('\nLoading model...')
    device = lib.get_device()
    model = Transformer(
        d_numerical=x_num.shape[1],
        categories=cat_values,
        d_out=1, ## regression hardcoded
        **args['model'],
    ).to(device)
    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    checkpoint_path = output / 'checkpoint.pt'
    model.load_state_dict(torch.load(checkpoint_path)['model'])

    ## (4) exemplary call, second test entry
    print('\nTest evaluation...')

    y=predict(x_num, x_cat, num_new_values, max_values, normalizer, encoder, y_mean_std, device)
    sys.exit()

    print(y)

