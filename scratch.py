import h5py
import numpy as np
import pytorch3d as torch3d
import open3d as o3d
import os
from scipy.spatial.transform.rotation import Rotation as R


### Constants and functions
SHAPENET_PART_DATA_ROOT = '/home/donglin/Data/shapenetcore_partanno_segmentation_benchmark_v0'
SHAPENETV1_MODEL_PATH_TEMPLATE = '/home/donglin/Data/ShapeNetCore.v1/{shapenet_cat_id}/{shapenet_id}/model.obj'
SHAPENETV2_MODEL_PATH_TEMPLATE = '/home/donglin/Data/ShapeNetCore.v2/{shapenet_cat_id}/{shapenet_id}/models/model_normalized.obj'
DATA_ROOT = '/home/donglin/Data/sem_seg_h5/'
CAT = 'Lamp'
LEVEL = '1'
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def read_data_paths(data_path_file):
    root = DATA_ROOT + CAT + '-' + LEVEL + '/'
    with open(data_path_file, 'r') as file:
        paths = file.readlines()
    return [root + path[2:] for path in paths]

def get_combined_data(paths):
    data = []
    label_seg = []
    for path in paths:
        f = h5py.File(path, 'r')
        data.append(np.array(f['data'][:]))
        label_seg.append(np.array(f['label_seg'][:]))
    return np.concatenate(data), np.concatenate(label_seg)

def get_icp_between_pointclouds(source, target, use_normal=False, threshold=1, trans_init=np.eye(4), max_iteration=2000):
    if use_normal:
        assert target.normals != None, "Target pointcloud must have normals."
    loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
    source_to_target = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return source_to_target.transformation

def get_min_max_center(pcd):
    minVertex = np.array([np.Infinity, np.Infinity, np.Infinity])
    maxVertex = np.array([-np.Infinity, -np.Infinity, -np.Infinity])
    aggVertices = np.zeros(3)
    numVertices = 0
    for pnt in pcd:
        aggVertices += pnt
        numVertices += 1
        minVertex = np.minimum(pnt, minVertex)
        maxVertex = np.maximum(pnt, maxVertex)
    centroid = aggVertices / numVertices
    info = {}
    info['min'] = minVertex
    info['max'] = maxVertex
    info['centroid'] = centroid
    return info


def normalize_points(pcd):
    result = []
    stats = get_min_max_center(pcd)
    diag = np.array(stats['max']) - np.array(stats['min'])
    norm = 1 / np.linalg.norm(diag)
    c = stats['centroid']
    for v in pcd:
        v_new = (v - c) * norm
        result.append(v_new)
    return np.stack(result), norm, c

def normalize_points(pcd):
    stats = get_min_max_center(pcd)
    diag = np.array(stats['max']) - np.array(stats['min'])
    norm = 1 / np.linalg.norm(diag)
    trans = np.eye(4)
    trans[:3,:3] *= norm
    trans[:3,3] = -stats['centroid']
    return trans

def load_obj(fn):
    # EXPERIMENTAL CITATION
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0]
                         for item in line.split()[1:4]]))
    return np.vstack(vertices), np.vstack(faces) - 1

def get_mesh_from_verices_and_faces(vertices, faces):
    # EXPERIMENTAL
    vertices, faces = o3d.utility.Vector3dVector(
        vertices), o3d.utility.Vector3iVector(faces)
    return o3d.geometry.TriangleMesh(vertices, faces)

### Testing the number of unique labels in the 'Lamp' class in Partnet

# label1 = h5py.File('/home/donglin/Data/sem_seg_h5/Lamp-3/train-00.h5', 'r')
# label1 = np.array(label1['label_seg'][:])
# label2 = h5py.File('/home/donglin/Data/sem_seg_h5/Lamp-3/train-01.h5', 'r')
# label2 = np.array(label2['label_seg'][:])
# label = np.concatenate((label1, label2))
# print(label.shape)
# print(np.unique(label))


### Visualizing the Shapenet Part dataset

## Creating Shapene Part category to cat id mapping
mapping = {}
cat_name_to_id = {}
cat_id_to_name = {}
with open(SHAPENET_PART_DATA_ROOT + '/' + 'synsetoffset2category.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        name, id = line.split('\t')
        name, id = name.strip(), id.strip()
        cat_id_to_name[id] = name
        cat_name_to_id[name] = id

# category = mapping['Mug']
category = '02954340'
shapenet_id = '5eb9ab53213f5fff4e09ebaf49b0cb2f'

## Creating point cloud vis
part_points_path = SHAPENET_PART_DATA_ROOT + '/' + category + '/points/' + shapenet_id + '.pts'
part_points = []
with open(part_points_path, 'r') as f:
    lines = f.readlines()
    for l in lines:
        l = l.strip()
        p = [float(token) for token in l.split(' ')]
        part_points.append(p)

x_rot = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()
y_rot = R.from_rotvec([0, 90, 0], degrees=True).as_matrix()
z_rot = R.from_rotvec([0, 0, 90], degrees=True).as_matrix()

trans_mtx = np.array([[0,0,1],[0,1,0],[-1,0,0]])
norm_mtx = normalize_points(part_points)
part_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(part_points)).rotate(trans_mtx).transform(norm_mtx)

## Creating mesh ShapenetCore v2
v2_path = SHAPENETV2_MODEL_PATH_TEMPLATE.format(shapenet_id=shapenet_id, shapenet_cat_id=category)
v2_mesh = o3d.io.read_triangle_mesh(v2_path)

## Checking whether the Shapenet v2 is normalized
# v2_mesh_norm = o3d.io.read_triangle_mesh(v2_path)
# v2_mesh_norm.vertices = o3d.utility.Vector3dVector(normalize_points(v2_mesh_norm.vertices))
# v2_mesh_norm.paint_uniform_color([0, 0, 1])
o3d.visualization.draw_geometries([part_points, v2_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame()])

## Shapenet Part to Shapenet v2 ICP
# v2_points = v2_mesh.sample_points_uniformly(8192, use_triangle_normal=True).normalize_normals()
# transformation = get_icp_between_pointclouds(part_points, v2_points)
# part_points = part_points.transform(transformation)
# o3d.visualization.draw_geometries([part_points, v2_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame()])


### Test generating processed Shapenet Part data
# transformations = []
# points = []
# for cat_id, cat in cat_id_to_name.items():
#     for filename in os.listdir(SHAPENET_PART_DATA_ROOT + '/' + cat_id + '/points'):
#         id = filename.split('.')[0]
#         part_path = SHAPENET_PART_DATA_ROOT + '/' + cat_id + '/points/' + filename
#         v2_path = SHAPENETV2_MODEL_PATH_TEMPLATE.format(shapenet_id=id, shapenet_cat_id=cat_id)
#         if not os.path.isfile(v2_path):
#             continue
#         verts, faces = load_obj(v2_path)
#         v2_mesh = get_mesh_from_verices_and_faces(verts, faces)
#         part_points = part_points @ trans_mtx.T
#         part_points = normalize_points(part_points)
#         part_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(part_points))
#         v2_points = v2_mesh.sample_points_uniformly(8192, use_triangle_normal=True).normalize_normals()
#         transformation = get_icp_between_pointclouds(part_points, v2_points)
#         part_points.transform(transformation)
#         points.append(np.array(part_points.points))
#         print(type(transformation))
    

### Test if loading obj manually is equal to the o3d way of loading
# verts, faces = load_obj('/home/donglin/Data/ShapeNetCore.v2/02691156/afa83b431ffe73a454eefcdc602d4520/models/model_normalized.obj')
# manual_mesh = get_mesh_from_verices_and_faces(verts, faces)
# auto_mesh = o3d.io.read_triangle_mesh('/home/donglin/Data/ShapeNetCore.v2/02691156/afa83b431ffe73a454eefcdc602d4520/models/model_normalized.obj')
