
import h5py
import numpy as np
import pytorch3d as torch3d
import open3d as o3d
import os
from utils import *
import csv
from scipy.spatial.transform.rotation import Rotation as R


### Constants and functions
ACRONYM_ROOT = '/home/donglin/Data/acronym'
SHAPENET_PART_DATA_ROOT = '/home/donglin/Data/shapenetcore_partanno_segmentation_benchmark_v0'
SHAPENETV1_ROOT_PATH = '/home/donglin/Data/ShapeNetCore.v1/'
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

def normalize_points_trans(pcd):
    stats = get_min_max_center(pcd)
    diag = np.array(stats['max']) - np.array(stats['min'])
    norm = 1 / np.linalg.norm(diag)
    trans = np.eye(4)
    trans[:3,:3] *= norm
    trans[:3,3] = -stats['centroid'] * norm
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


def normalize_mesh(mesh):
    vertices, _, _ = normalize_points(np.asarray(mesh.vertices))
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


def read_shapenetpart_points(path):
    points = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip()
            p = [float(token) for token in l.split(' ')]
            points.append(p)

    return np.stack(points)


def axis_to_so3(up, front):
    trans = np.eye(3)
    up = np.array(up, dtype=np.float64) 
    front = np.array(front, dtype=np.float64)
    rear = np.cross(up, front)
    
    trans[:, 0] = front
    trans[:, 1] = rear
    trans[:, 2] = up
    return np.linalg.inv(trans)

def normalize_pointcloud(pcd):
    points, _, _ = normalize_points(np.asarray(pcd.points))
    pcd.points = o3d.utility.Vector3iVector(points)
    return pcd


def get_normalized_pointcloud():
    
    pass


if __name__ == '__main__':

    # Testing the number of unique labels in the 'Lamp' class in Partnet

    # label1 = h5py.File('/home/donglin/Data/sem_seg_h5/Lamp-3/train-00.h5', 'r')
    # label1 = np.array(label1['label_seg'][:])
    # label2 = h5py.File('/home/donglin/Data/sem_seg_h5/Lamp-3/train-01.h5', 'r')
    # label2 = np.array(label2['label_seg'][:])
    # label = np.concatenate((label1, label2))
    # print(label.shape)
    # print(np.unique(label))

    # Visualizing the Shapenet Part dataset

    def create_shapenet_part_cat_name_id_map():
        # Creating Shapenet Part category to cat id mapping
        cat_name_to_id = {}
        cat_id_to_name = {}
        with open(SHAPENET_PART_DATA_ROOT + '/' + 'synsetoffset2category.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                name, id = line.split('\t')
                name, id = name.strip(), id.strip()
                cat_id_to_name[id] = name
                cat_name_to_id[name] = id
        return cat_name_to_id, cat_id_to_name

    def visualize_shapenet_part_to_v2():
        category = '02954340'
        shapenet_id = '5eb9ab53213f5fff4e09ebaf49b0cb2f'

        # Creating point cloud vis
        part_points_path = SHAPENET_PART_DATA_ROOT + '/' + \
            category + '/points/' + shapenet_id + '.pts'
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

        trans_mtx = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        norm_mtx = normalize_points(part_points)
        part_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            part_points)).rotate(trans_mtx).transform(norm_mtx)

        # Creating Shapenet Sem Mesh
        shapenet_id = 'b88bcf33f25c6cb15b4f129f868dedb'

        # Creating mesh ShapenetCore v2
        v2_path = SHAPENETV2_MODEL_PATH_TEMPLATE.format(
            shapenet_id=shapenet_id, shapenet_cat_id=category)
        v2_mesh = o3d.io.read_triangle_mesh(v2_path)
        o3d.visualization.draw_geometries(
            [part_points, v2_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame()])

    # Checking whether the Shapenet v2 is normalized
    # v2_mesh_norm = o3d.io.read_triangle_mesh(v2_path)
    # v2_mesh_norm.vertices = o3d.utility.Vector3dVector(normalize_points(v2_mesh_norm.vertices))
    # v2_mesh_norm.paint_uniform_color([0, 0, 1])

    # Shapenet Part to Shapenet v2 ICP
    # v2_points = v2_mesh.sample_points_uniformly(8192, use_triangle_normal=True).normalize_normals()
    # transformation = get_icp_between_pointclouds(part_points, v2_points)
    # part_points = part_points.transform(transformation)
    # o3d.visualization.draw_geometries([part_points, v2_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame()])

    # Test generating processed Shapenet Part data
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

    # Test if loading obj manually is equal to the o3d way of loading
    # verts, faces = load_obj('/home/donglin/Data/ShapeNetCore.v2/02691156/afa83b431ffe73a454eefcdc602d4520/models/model_normalized.obj')
    # manual_mesh = get_mesh_from_verices_and_faces(verts, faces)
    # auto_mesh = o3d.io.read_triangle_mesh('/home/donglin/Data/ShapeNetCore.v2/02691156/afa83b431ffe73a454eefcdc602d4520/models/model_normalized.obj')

    SHAPENET_SEM_ROOT = '/home/donglin/Data/ShapeNetSem.v0/'

    def test_sem_shapenet_part_overlap():
        pass

    def test_sem_to_v2_alignment1():
        # sem_cat_id = '3802912'
        # v2_cat_id = '03797390'
        # id = '10f6e09036350e92b3f21f1137c3c347' # This is the 56th row of the ShapeNetSem metadata

        sem_cat_id = '3802912'
        v2_cat_id = '03797390'
        id = '128ecbc10df5b05d96eaf1340564a4de'

        sem_path = os.path.join(
            SHAPENET_SEM_ROOT, 'models-OBJ', 'models', f'{id}.obj')
        sem_mesh = normalize_mesh(o3d.io.read_triangle_mesh(
            sem_path)).compute_triangle_normals()
        # x_rot = R.from_rotvec([-90, 0, 0], degrees=True).as_matrix()
        # y_rot = R.from_rotvec([0, 180, 0], degrees=True).as_matrix()
        # sem_mesh.rotate(x_rot)
        # sem_mesh.rotate(y_rot)

        v2_path = SHAPENETV2_MODEL_PATH_TEMPLATE.format(
            shapenet_id=id, shapenet_cat_id=v2_cat_id)
        v2_mesh = o3d.io.read_triangle_mesh(v2_path).paint_uniform_color(
            [1, 0, 0]).compute_triangle_normals()
        sem_pc = sem_mesh.sample_points_uniformly(1000)
        v2_pc = v2_mesh.sample_points_uniformly(1000)
        trans = get_icp_between_pointclouds(
            sem_pc, v2_pc, use_normal=True, max_iteration=20000)
        # sem_mesh.transform(trans)

        o3d.visualization.draw_geometries(
            [sem_mesh, v2_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame()])

    def test_sem_alignment():
        sem_id1 = '128ecbc10df5b05d96eaf1340564a4de'
        # sem_id2 = '159e56c18906830278d8f8c02c47cde0'
        # sem_id2 = '187859d3c3a2fd23f54e1b6f41fdd78a'
        # sem_id2 = '6aec84952a5ffcf33f60d03e1cb068dc'
        # sem_id2 = '187859d3c3a2fd23f54e1b6f41fdd78a'
        sem_id2 = '10f6e09036350e92b3f21f1137c3c347'
        sem_path1 = os.path.join(
            SHAPENET_SEM_ROOT, 'models-OBJ', 'models', f'{sem_id1}.obj')
        sem_path2 = os.path.join(
            SHAPENET_SEM_ROOT, 'models-OBJ', 'models', f'{sem_id2}.obj')
        mesh1 = normalize_mesh(o3d.io.read_triangle_mesh(
            sem_path1).paint_uniform_color([1, 0, 0])).compute_triangle_normals()
        mesh2 = normalize_mesh(o3d.io.read_triangle_mesh(
            sem_path2).paint_uniform_color([0, 1, 0])).compute_triangle_normals()
        o3d.visualization.draw_geometries([mesh1, mesh2, o3d.geometry.TriangleMesh.create_coordinate_frame()])

    def test_sem_v2_regis():
        sem_id1 = '128ecbc10df5b05d96eaf1340564a4de'
        sem_id2 = '10f6e09036350e92b3f21f1137c3c347'
        sem_path1 = os.path.join(
            SHAPENET_SEM_ROOT, 'models-OBJ', 'models', f'{sem_id1}.obj')
        sem_path2 = os.path.join(
            SHAPENET_SEM_ROOT, 'models-OBJ', 'models', f'{sem_id2}.obj')
        mesh1 = normalize_mesh(o3d.io.read_triangle_mesh(
            sem_path1).paint_uniform_color([1, 0, 0])).compute_triangle_normals()
        mesh2 = normalize_mesh(o3d.io.read_triangle_mesh(
            sem_path2).paint_uniform_color([0, 1, 0])).compute_triangle_normals()
        pc1 = mesh1.sample_points_uniformly(10000)
        pc2 = mesh2.sample_points_uniformly(10000)
        trans = get_icp_between_pointclouds(pc1, pc2)
        mesh1.transform(trans)
        o3d.visualization.draw_geometries([mesh1, mesh2])
        pass

    def test_sem_shapenetpart_reg():
        # shape_id = '10f6e09036350e92b3f21f1137c3c347'
        shape_id = '128ecbc10df5b05d96eaf1340564a4de'
        part_cat_id = '03797390'
        sem_path = os.path.join(
            SHAPENET_SEM_ROOT, 'models-OBJ', 'models', f'{shape_id}.obj')
        sem_mesh = normalize_mesh(o3d.io.read_triangle_mesh(sem_path))
        rot = axis_to_so3([0,0,1],[1,0,0])
        sem_mesh.rotate(rot)
        x_rot = R.from_rotvec([-90, 0, 0], degrees=True).as_matrix()
        sem_mesh.rotate(x_rot)
        shapenetpart_path = os.path.join(
            SHAPENET_PART_DATA_ROOT, part_cat_id, 'points', f'{shape_id}.pts')
        shapenetpart_pnt = o3d.utility.Vector3dVector(
            read_shapenetpart_points(shapenetpart_path))
        shapenetpart_pcd = o3d.geometry.PointCloud()
        shapenetpart_pcd.points = shapenetpart_pnt
        shapenetpart_pcd = normalize_pointcloud(shapenetpart_pcd)
        o3d.visualization.draw_geometries(
            [sem_mesh, shapenetpart_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame()])
        pass

    def test_sem_v1_alig():
        shapenet_id = '10f6e09036350e92b3f21f1137c3c347'
        cat_id = '03797390'
        v1_meta_path = os.path.join(SHAPENETV1_ROOT_PATH, f'{cat_id}.csv')
        v1_meta = {}
        with open(v1_meta_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row['fullId']:
                    continue
                cur_id = row['fullId'][5:]
                cur_meta = {}
                cur_meta['up'] = [float(ele) for ele in row['up'].split(
                    '\,')] if row['up'] else None
                cur_meta['front'] = [float(ele) for ele in row['front'].split(
                    '\,')] if row['front'] else None
                v1_meta[cur_id] = cur_meta
        print(v1_meta)
        pass

    def index_v1():
        v1_meta = {}
        for dir in os.listdir(SHAPENETV1_ROOT_PATH):
            if not dir.endswith('.csv'):
                continue

            with open(os.path.join(SHAPENETV1_ROOT_PATH, dir), 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row['fullId']:
                        continue
                    cur_id = row['fullId'][4:]
                    cur_meta = {}
                    cur_meta['up'] = [float(ele) for ele in row['up'].split(
                        '\,')] if row['up'] else None
                    cur_meta['front'] = [float(ele) for ele in row['front'].split(
                        '\,')] if row['front'] else None
                    cur_meta['cat_id'] = dir[:-4]
                    cur_meta['path'] = os.path.join(
                        SHAPENETV1_ROOT_PATH, dir[:-4], cur_id)
                    v1_meta[cur_id] = cur_meta
        return v1_meta

    def index_sem():
        sem_meta = {}
        sem_path = os.path.join(SHAPENET_SEM_ROOT, 'models-OBJ', 'models')
        for dir in os.listdir(sem_path):
            if dir.startswith('room'):
                continue
            if dir.endswith('.mtl'):
                continue
            cur_meta = {}
            cur_id = dir[:-4]
            cur_meta['path'] = os.path.join(sem_path, dir)
            sem_meta[cur_id] = cur_meta
        return sem_meta

    def index_acronym():
        grasp_path = os.path.join(ACRONYM_ROOT, 'grasps')
        acro_meta = {}
        for dir in os.listdir(grasp_path):
            cat_name, shape_id, scale = dir.split('_')
            scale = float(scale[:-3])
            cur_meta = {}
            cur_meta['cat_name'] = cat_name
            cur_meta['path'] = os.path.join(grasp_path, dir)
            cur_meta['scale'] = scale
            acro_meta[shape_id] = cur_meta
        return acro_meta

    def index_shapenetpart():
        pass


    def test_calc_sem_to_shapenetpart_alignment(sem_meta, v1_meta):
        for id, info in sem_meta.items():
            if id != '10f6e09036350e92b3f21f1137c3c347':
                continue
            if not id in sem_meta or not id in v1_meta:
                continue
            sem_mesh = o3d.io.read_triangle_mesh(info['path']).paint_uniform_color(
                [1, 0, 0]).compute_triangle_normals()

            norm_trans = normalize_points_trans(np.asarray(sem_mesh.vertices))
            rot1 = so3_to_se3(axis_to_so3(v1_meta[id]['up'], v1_meta[id]['front']))
            rot2 = so3_to_se3(R.from_rotvec([-90, 0, 0], degrees=True).as_matrix())
            rot = rot2 @ rot1 @ norm_trans
            sem_mesh.transform(rot)
      
            shapenetpart_path = os.path.join(
            SHAPENET_PART_DATA_ROOT, v1_meta[id]['cat_id'], 'points', f'{id}.pts')
            shapenetpart_pnt, _, _ = normalize_points(read_shapenetpart_points(shapenetpart_path))
            shapenetpart_pcd = o3d.geometry.PointCloud()
            shapenetpart_pcd.points = o3d.utility.Vector3dVector(shapenetpart_pnt)
            o3d.visualization.draw_geometries(
            [sem_mesh, shapenetpart_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame()])
            break

    def visualize_acro_scale_with_sem(sem_meta, v1_meta, acro_meta):
        shape_id = '57f73714cbc425e44ae022a8f6e258a7'
        info = sem_meta[shape_id]
        sem_mesh = o3d.io.read_triangle_mesh(info['path']).paint_uniform_color(
                [1, 0, 0]).compute_triangle_normals()
        norm_trans = normalize_points_trans(np.asarray(sem_mesh.vertices))
        rot1 = so3_to_se3(axis_to_so3(v1_meta[shape_id]['up'], v1_meta[shape_id]['front']))
        rot2 = so3_to_se3(R.from_rotvec([-90, 0, 0], degrees=True).as_matrix())
        rot = rot2 @ rot1 @ norm_trans
        sem_mesh.transform(rot)
        
        acro_mesh = o3d.io.read_triangle_mesh(info['path']).paint_uniform_color(
                [0, 1, 0]).compute_triangle_normals()
        scale = acro_meta[shape_id]['scale']
        acro_mesh.scale(scale, center=acro_mesh.get_center())
        o3d.visualization.draw_geometries(
            [sem_mesh, acro_mesh, o3d.geometry.TriangleMesh.create_coordinate_frame()])

        
    def get_acro_success_grasps(shape_id, acronym_index, sem_index):
        vis = []
        grasp_file = h5py.File(acronym_index[shape_id]['path'])
        grasps = np.array(grasp_file['grasps/transforms'])
        for trans in grasps[:3]:
            gripper_mesh = o3d.io.read_triangle_mesh('./models/franka_gripper_collision_mesh.stl').compute_triangle_normals()
            gripper_mesh.transform(trans)
            vis.append(gripper_mesh)
        acro_mesh = o3d.io.read_triangle_mesh(sem_index[shape_id]['path']).paint_uniform_color(
                [0, 1, 0]).compute_triangle_normals()
        scale = acronym_index[shape_id]['scale']
        acro_mesh.scale(scale, center=[0,0,0])
        
        acro_mesh2 = o3d.io.read_triangle_mesh('/home/donglin/Data/acronym/meshes/Mug/' + f'{shape_id}.obj')
        acro_mesh2.scale(scale, center=[0,0,0])


        vis.append(acro_mesh)
        # vis.append(acro_mesh2)
        vis.append(o3d.geometry.TriangleMesh.create_coordinate_frame())
        o3d.visualization.draw_geometries(vis)
        pass

    def get_bounding_boxes():
        pass


    # test_sem_to_v2_alignment1()
    # test_sem_alignment()
    # test_sem_v2_regis()
    # test_sem_shapenetpart_reg()
    # test_sem_v1_alig()
    # visualize_acro_scale_with_sem(index_sem(), index_v1(), index_acronym())
    get_acro_success_grasps('57f73714cbc425e44ae022a8f6e258a7', index_acronym(), index_sem())
    # test_calc_sem_to_shapenetpart_alignment(b, a)