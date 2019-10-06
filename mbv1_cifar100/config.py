import sys

## default configuration for mobilenetv1 ##
defaultcfg = [
    32, 64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024,
]

splitcfg = [
    8, 8, (8, 2), 8, (16, 2), 16, (24, 2), 24, 24, 24, 24, 24, (32, 2), 32,
]

## maximum #neurons ##
max_splitcfg = [
    32, 64, (128, 2), 128, (320, 2), 320, (640, 2), 640, 640, 640, 640, 640, (1280, 2), 1280,
]

split_groups = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

## out resolutions and kernel size for each filter ##
resolutions = [
    (32, 3), (32, 1), (16, 1), (16, 1), (8, 1), (8, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (4, 1), (2, 1), (2, 1),
]

get_number_of_channels = lambda x: x if isinstance(x, int) else x[0]

def get_flops_inc_per_layer(model, layer_idx):

    out_res, kernel_size = resolutions[layer_idx]
    pre_nc = 3 if layer_idx == 0 else get_number_of_channels(model.cfg[layer_idx-1])
    kernel_ops = kernel_size * kernel_size * pre_nc
    
    flops_inc_before = kernel_ops * out_res * out_res
    if layer_idx > 0: 
        depth_conv_ops = 3 * 3 * out_res * out_res
        flops_inc_before += depth_conv_ops
    
    # if add one neuron, how many more flops for the next layer
    if layer_idx < len(resolutions) - 1:
        #next_nc = get_number_of_channels(cfg[i+1]) # if not depth wise
        next_nc = 1
        next_res, next_k = resolutions[layer_idx+1]
        flops_inc_after = next_nc * (next_res**2) * (next_k**2)
    else:
        #flops_inc_after = model.num_classes
        flops_inc_after = model.num_classes # as a constant
    flops_inc = flops_inc_before + flops_inc_after
    #print(layer_idx, flops_inc)
    return flops_inc


def get_params_inc_per_layer(model, layer_idx):
    out_res, kernel_size = resolutions[layer_idx]
    pre_nc = 3 if layer_idx == 0 else get_number_of_channels(model.cfg[layer_idx-1])

    params_inc = pre_nc * kernel_size * kernel_size
    return params_inc




