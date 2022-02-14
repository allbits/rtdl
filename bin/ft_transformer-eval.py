# %%
import sys
import os
import pickle
import numpy as np
import zero

import lib
from ft_transformer import *
import evaluation

# %%
if __name__ == "__main__":
    ## (1) load the configuration, general setup
    args, output = lib.load_config()
    args['model'].setdefault('token_bias', True)
    args['model'].setdefault('kv_compression', None)
    args['model'].setdefault('kv_compression_sharing', None)

    dataset_dir = lib.get_path(args['data']['path'])
    dataset_info = lib.load_json(dataset_dir / 'info.json')

    seed=args['seed']
    zero.set_randomness(seed)

    ## (2) load the transformation data
    tf = evaluation.Normalization(args,seed)

    ## (3) example test data
    if dataset_info['n_num_features'] == 8:
        print("Assuming ACOTSP dataset.")
        x_num=[1.83, 7.87, 0.69, 36.0, np.nan, np.nan, np.nan, 6.0 ]
        x_cat=['129', 'as', '2', 'nan' ]
    else:
        print("Assuming LKH dataset.")
        x_num=[255.0, 0.0, 5.0, 5.0, 4.0, 3.0, 12.0, 14.0, 20.0, 5.0, 986.0, 5.0]
        x_cat=['121', 'NO', 'QUADRANT', 'QUADRANT', 'YES', 'YES', 'GREEDY', 'NO', 'NO', 'YES']

    x_num=np.array(x_num).reshape(1,-1)

    ## (4) load the model (and possibly move it to the GPU)
    print(f'\nLoading model ({x_num.shape[1]}n/{len(tf.cat_values)}c) ...')
    device = lib.get_device()
    model = Transformer(
        d_numerical=x_num.shape[1],
        categories=tf.cat_values,
        d_out=1, ## regression hardcoded
        **args['model'],
    ).to(device)
    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    checkpoint_path = output / 'checkpoint.pt'
    model.load_state_dict(torch.load(checkpoint_path)['model'])

    ## (5) exemplary call
    print('\nTest evaluation...')
    x_num, x_cat = tf.normalize_x(x_num, x_cat)
    with torch.no_grad():
        if device.type != 'cpu':
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
        y_raw = model(x_num,x_cat)
    print(tf.normalize_y(y_raw))
