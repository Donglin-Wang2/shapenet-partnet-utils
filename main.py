from shapenet_v2 import *
from shapenet_part import *

if __name__ == '__main__':
    ## Testing for Shapenet v2
    shapenet_v2 = ShapenetV2()
    shapenet_v2.gen_record()
    ## Testing Shapenet Part
    shapenet_part = ShapenetPart()
    shapenet_part.get_records()
    # shapenet_part.register_points()