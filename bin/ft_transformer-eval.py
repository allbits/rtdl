# %%
from . import ft_transformer
import lib

# %%
if __name__ == "__main__":

    ## (1) load the configuration, general setup
    args, output = lib.load_config()
    args['model'].setdefault('token_bias', True)
    args['model'].setdefault('kv_compression', None)
    args['model'].setdefault('kv_compression_sharing', None)
    zero.set_randomness(args['seed'])

    ## (2) load the transformation data
    ## https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/
    ## somewhere in the original code: dump(normalizer, open('normalizer.pkl', 'wb'))

    ## num_new_values
    normalizer = load(open('normalizer.pkl', 'rb'))
    encoder = load(open('encoder.pkl', 'rb'))
    ## max_values
    ## y_std, y_mean
    
    ## (3) load the model (and possibly move it to the GPU)
    print('Loading model...')
    model = Transformer(
        d_numerical=0 if X_num is None else X_num['train'].shape[1],
        categories=lib.get_categories(X_cat),
        d_out=D.info['n_classes'] if D.is_multiclass else 1,
        **args['model'],
    ).to(device)
    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    
    ## (4) exemplary call, second test entry
    x_num=[1.83, 7.87, 0.69, 36.0, nan, nan, nan, 6.0 ]
    x_cat=['129', 'as', '2', '1']

    device = lib.get_device()
    if device.type != 'cpu':
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        
    y=predict(x_num,x_cat, num_new_values, max_values, cat_policy, normalizer, encoder)

    print(y)

    ## predict a value
    ##   Assumes:
    ##     num_nan_policy=='mean'
    ##     cat_nan_policy=='new'
    ##     cat_policy=='indices'
    @torch.no_grad()
    def predict(x_num, x_cat, num_new_values, max_values, normalizer, encoder):
        ## (4.1) transform numerical data
        ## (4.1.1) replace nan by mean
        num_nan_mask = np.isnan(x_num)
        if num_nan_mask.any():
            num_nan_indices = np.where(num_nan_mask)
            x_num[num_nan_indices] = np.take(num_new_values, num_nan_indices)
        
        ## (4.1.2) normalize
        x_num = normalizer.transform(x_num)
        x_num = torch.as_tensor(x_num)
        
        ## (4.2) transform categorical data
        ## (4.2.1) replace nan
        cat_nan_mask = x_cat == 'nan'

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
        y_raw = model(x_num,x_cat)
        
        ## (4.4) transform the output back
        return (y_raw*y_std)+y_mean
