import json

from shapenet_v2 import *
from shapenet_part import *
from shapenet_sem import *
from acronym import *

if __name__ == '__main__':
    '''Testing for Shapenet v2'''
    shapenet_v2 = ShapenetV2()
    shapenet_v2.gen_record()
    '''Testing Shapenet Part'''
    # shapenet_part = ShapenetPart()
    # shapenet_part.gen_records()
    # shapenet_part.register_data()
    # shapenet_part.write_data(register=True)
    '''Testing point registration'''
    # shapenet_part = ShapenetPart()
    # count = 0
    # for id, record in shapenet_part.records.content.items():
    #     if count == 10:
    #         break
    #     if record.part_info and record.v2_info:
    #         points = shapenet_part.read_data(record.part_info.path)
    #         points = get_registered_pointcloud(points, record.part_info.registry)
    #         verts, faces = read_obj(record.v2_info.path)
    #         v2_mesh = get_mesh_from_verices_and_faces(verts, faces)
    #         o3d.visualization.draw_geometries([points, v2_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame()])
    #         count += 1
    # pass
    '''Testing Acronym'''
    # acronym = Acronym()
    # acronym.gen_record()

    '''Testing Shapenet Sem'''
    shapenet_sem = ShapenetSem()
    shapenet_sem.gen_record()