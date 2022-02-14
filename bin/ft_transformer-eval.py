# %%
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

    seed=args['seed']
    zero.set_randomness(seed)

    ## (2) load the transformation data, get example data
    tf = evaluation.Normalization(args,seed)

    ## (3) load the model (and possibly move it to the GPU)
    print(f'\nLoading model ({tf.n_num_features}n/{tf.n_cat_features}c) ...')
    device = lib.get_device()
    model = Transformer(d_numerical=tf.n_num_features, categories=tf.cat_values,
        d_out=1, ## regression hardcoded
        **args['model'],
    ).to(device)
    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    checkpoint_path = output / 'checkpoint.pt'
    model.load_state_dict(torch.load(checkpoint_path)['model'])

    ## (4) exemplary call
    print('\nTest evaluation...')
    x_num,x_cat = tf.get_example_data()
    x_num,x_cat = tf.normalize_x(x_num, x_cat)
    with torch.no_grad():
        if device.type != 'cpu':
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
        y_raw = model(x_num,x_cat)
    print(tf.normalize_y(y_raw))
