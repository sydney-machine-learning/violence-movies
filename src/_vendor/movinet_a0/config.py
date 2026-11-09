"""
Inspired by
https://github.com/PeizeSun/SparseR-CNN/blob/dff4c43a9526a6d0d2480abc833e78a7c29ddb1a/detectron2/config/defaults.py
"""
from types import SimpleNamespace as CN

def fill_SE_config(conf, input_channels, 
                    out_channels, 
                    expanded_channels,
                    kernel_size,
                    stride,
                    padding,
                    padding_avg,
):
    conf.expanded_channels =expanded_channels
    conf.padding_avg= padding_avg
    fill_conv(conf,input_channels,
                out_channels, 
                kernel_size,
                stride,
                padding,
)

def fill_conv(conf, input_channels,
                out_channels, 
                kernel_size,
                stride,
                padding,):
    conf.input_channels = input_channels
    conf.out_channels = out_channels
    conf.kernel_size = kernel_size
    conf.stride = stride
    conf.padding = padding
   

    

_C = CN()

_C.MODEL = CN()


###################
#### MoViNetA0 ####
###################

_C.MODEL.MoViNetA0 = CN()
_C.MODEL.MoViNetA0.name = "A0"
_C.MODEL.MoViNetA0.weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA0_statedict_v3?raw=true"
_C.MODEL.MoViNetA0.stream_weights = "https://github.com/Atze00/MoViNet-pytorch/blob/main/weights/modelA0_stream_statedict_v3?raw=true"
_C.MODEL.MoViNetA0.conv1 = CN()
fill_conv(_C.MODEL.MoViNetA0.conv1, 3,8,(1,3,3),(1,2,2),(0,1,1))


_C.MODEL.MoViNetA0.blocks = [ [CN()],
                            [CN() for _ in range(3)],
                            [CN() for _ in range(3)],
                            [CN() for _ in range(4)],
                            [CN() for _ in range(4)]]

#Block2
fill_SE_config(_C.MODEL.MoViNetA0.blocks[0][0], 8, 8, 24, (1,5,5), (1,2,2), (0,2,2), (0,1,1))

#block 3
fill_SE_config(_C.MODEL.MoViNetA0.blocks[1][0], 8, 32, 80, (3,3,3), (1,2,2), (1,0,0), (0,0,0))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[1][1], 32, 32, 80, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[1][2], 32, 32, 80, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 4
fill_SE_config(_C.MODEL.MoViNetA0.blocks[2][0], 32, 56, 184, (5,3,3), (1,2,2), (2,0,0), (0,0,0))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[2][1], 56, 56, 112, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[2][2], 56, 56, 184, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 5
fill_SE_config(_C.MODEL.MoViNetA0.blocks[3][0], 56, 56, 184, (5,3,3), (1,1,1), (2,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[3][1], 56, 56, 184, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[3][2], 56, 56, 184, (3,3,3), (1,1,1), (1,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[3][3], 56, 56, 184, (3,3,3), (1,1,1), (1,1,1), (0,1,1))

#block 6
fill_SE_config(_C.MODEL.MoViNetA0.blocks[4][0], 56, 104, 384, (5,3,3), (1,2,2), (2,1,1), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[4][1], 104, 104, 280, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[4][2], 104, 104, 280, (1,5,5), (1,1,1), (0,2,2), (0,1,1))
fill_SE_config(_C.MODEL.MoViNetA0.blocks[4][3], 104, 104, 344, (1,5,5), (1,1,1), (0,2,2), (0,1,1))

_C.MODEL.MoViNetA0.conv7= CN()
fill_conv(_C.MODEL.MoViNetA0.conv7, 104,480,(1,1,1),(1,1,1),(0,0,0))

_C.MODEL.MoViNetA0.dense9= CN()
_C.MODEL.MoViNetA0.dense9.hidden_dim = 2048



###################
#### MoViNetA1 ####
###################

