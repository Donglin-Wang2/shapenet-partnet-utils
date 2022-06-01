import json

from shapenet_v2 import *
from shapenet_part import *

if __name__ == '__main__':
    # Testing for Shapenet v2
    # shapenet_v2 = ShapenetV2()
    # shapenet_v2.gen_record()
    # Testing Shapenet Part
    shapenet_part = ShapenetPart()
    # shapenet_part.gen_records()
    # shapenet_part.register_data()
    shapenet_part.write_data(register=True)
    # for id in shapenet_part.records.content.keys():
    #     if hasattr(shapenet_part.records.content[id], 'part_info'):
    #         assert shapenet_part.records.content[id].registry.shape == (4, 4)
