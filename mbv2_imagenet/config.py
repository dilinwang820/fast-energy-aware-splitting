import sys

# output channels,  stride, skip connection
defaultcfg = [
    (32, 2, 0), # first conv
    (16, 1, 0), # first mb block
    (96, 1, 0), (24, 2, 0), 
    (144, 1, 1), (24, 1, 1), 
    (144, 1, 0), (32, 2, 0), 
    (192, 1, 1), (32, 1, 1), 
    (192, 1, 1), (32, 1, 1), 
    (192, 1, 0), (64, 2, 0), 
    (384, 1, 1), (64, 1, 1), 
    (384, 1, 1), (64, 1, 1), 
    (384, 1, 1), (64, 1, 1), 
    (384, 1, 0), (96, 1, 0), 
    (576, 1, 1), (96, 1, 1), 
    (576, 1, 1), (96, 1, 1), 
    (576, 1, 0), (160, 2, 0), 
    (960, 1, 1), (160, 1, 1), 
    (960, 1, 1), (160, 1, 1), 
    (960, 1, 0), (320, 1, 0), 
    (1280, 1, 0) # last conv
]

# output channels,  stride, skip connection
max_splitcfg = [
    (32, 1, 0), # first conv
    (20, 1, 0), # first mb block
    (120, 1, 0), (30, 1, 0), 
    (180, 1, 1), (30, 1, 1), 
    (180, 1, 0), (40, 2, 0), 
    (240, 1, 1), (40, 1, 1), 
    (240, 1, 1), (40, 1, 1), 
    (240, 1, 0), (80, 2, 0), 
    (480, 1, 1), (80, 1, 1), 
    (480, 1, 1), (80, 1, 1), 
    (480, 1, 1), (80, 1, 1), 
    (480, 1, 0), (120, 1, 0), 
    (720, 1, 1), (120, 1, 1), 
    (720, 1, 1), (120, 1, 1), 
    (720, 1, 0), (200, 2, 0), 
    (1200, 1, 1), (200, 1, 1), 
    (1200, 1, 1), (200, 1, 1), 
    (1200, 1, 0), (400, 1, 0), 
    (1600, 1, 0) # last conv
]

splitcfg = [
    (11, 2, 0), # first conv
    (6, 1, 0),  # first mb block, without expand 
    (30, 1, 0), (8, 2, 0),
    (40, 1, 1), (8, 1, 1),
    (40, 1, 0), (11, 2, 0),
    (55, 1, 1), (11, 1, 1),
    (55, 1, 1), (11, 1, 1),
    (55, 1, 0), (22, 2, 0),
    (110, 1, 1), (22, 1, 1),
    (110, 1, 1), (22, 1, 1),
    (100, 1, 1), (22, 1, 1),
    (110, 1, 0), (33, 1, 0),
    (165, 1, 1), (33, 1, 1),
    (165, 1, 1), (33, 1, 1),
    (165, 1, 0), (56, 2, 0),
    (280, 1, 1), (56, 1, 1),
    (280, 1, 1), (56, 1, 1),
    (280, 1, 0), (112, 1, 0),
    (448, 1, 0) # last conv
]

resolutions = [
    (112, 3), # first conv
    (112, 1), 
    (112, 1), (56, 1), 
    (56, 1), (56, 1), 
    (56, 1), (28, 1), 
    (28, 1), (28, 1), 
    (28, 1), (28, 1), 
    (28, 1), (14, 1), 
    (14, 1), (14, 1), 
    (14, 1), (14, 1), 
    (14, 1), (14, 1), 
    (14, 1), (14, 1), 
    (14, 1), (14, 1), 
    (14, 1), (14, 1), 
    (14, 1), (7, 1), 
    (7, 1), (7, 1), 
    (7, 1), (7, 1), 
    (7, 1), (7, 1), 
    (7, 1) # last conv
]

split_groups = [
    1,   
    0,
    1, 0,
    1, 0,
    1, 0,
    1, 0,
    1, 0, 
    1, 0,
    1, 0,
    1, 0,
    1, 0,
    1, 0,
    1, 0,
    1, 0,
    1, 0, 
    1, 0,
    1, 0,
    1, 0,
    0,
]


get_number_of_channels = lambda x: x[0]

def print_cfg(cfg):
    print('-------------------------------')
    out = []
    out.append(cfg[0][0])
    out.append(cfg[1][0])
    mb_cfg = cfg[2:-1]
    input_channel = cfg[1][0]
    for i in range(len(mb_cfg)//2):
        out.append((input_channel, mb_cfg[2*i][0], mb_cfg[2*i+1][0], 'skip' if mb_cfg[2*i+1][-1] else 'no_skip'))
        input_channel = mb_cfg[2*i+1][0]
    out.append(cfg[-1][0])
    print(out)
    print('-------------------------------')


def get_flops_inc_per_layer(model, layer_idx):

    out_res, kernel_size = resolutions[layer_idx]
    pre_nc = 3 if layer_idx == 0 else get_number_of_channels(model.cfg[layer_idx-1])
    kernel_ops = kernel_size * kernel_size * pre_nc

    flops_inc_before = kernel_ops * out_res * out_res
    # not the first or last conv
    if layer_idx > 0 and layer_idx < len(resolutions) - 1:
        pass # for simplicty

    # if add one neuron, how many more flops for the next layer
    if layer_idx < len(resolutions) - 1:
        # first mb block, layer_idx = 0
        if (layer_idx - 2) % 2 == 0:
            # top pw conv
            next_nc = 1  # next layer is depthwise
        else:
            # bottom pw conv
            next_nc = get_number_of_channels(model.cfg[layer_idx+1]) # not depthwise
        next_res, next_k = resolutions[layer_idx+1]
        flops_inc_after = next_nc * (next_res**2) * (next_k**2)
    else:
        #flops_inc_after = model.num_classes
        flops_inc_after = model.num_classes # as a constant
    flops_inc = flops_inc_before + flops_inc_after
    #print(layer_idx, flops_inc)
    return flops_inc


def get_params_inc_per_layer(model, layer_idx):
    return 1.0
    #out_res, kernel_size = resolutions[layer_idx]
    #pre_nc = 3 if layer_idx == 0 else get_number_of_channels(model.cfg[layer_idx-1])

    #params_inc = pre_nc * kernel_size * kernel_size
    #return params_inc


