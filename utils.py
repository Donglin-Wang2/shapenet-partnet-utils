import json
from typing import Optional
from configparser import ConfigParser

import numpy as np
import open3d as o3d

config = ConfigParser()
config.read('config.ini')


class Registry:
    def __init__(self):
        self.combined: np.ndarray = None
        self.norm: np.ndarray = None
        self.align: np.ndarray = None
        self.reg: np.ndarray = None
        self.loss: float = None

    def __str__(self):
        return obj_to_str(self)


class ItemInfo:
    def __init__(self):
        self.path: str = ''
        self.cat_id: Optional[str] = ''
        self.cat_name: Optional[str] = ''
        self.meta: Optional[str] = ''
        self.registry: Optional[np.ndarray] = None

    def __str__(self):
        return obj_to_str(self)


class DatasetInfo:
    def __init__(self):
        self.path: str = ''
        self.complete: bool = False
        self.meta: Optional[object] = {}

    def __str__(self):
        return obj_to_str(self)


class Record:
    def __init__(self):
        self.id: str = ''
        self.v1_info: Optional[ItemInfo] = {}
        self.v2_info: Optional[ItemInfo] = {}
        self.part_info: Optional[ItemInfo] = {}
        self.partnet_info: Optional[ItemInfo] = {}
        self.sem_info: Optional[ItemInfo] = {}

    def __str__(self):
        return obj_to_str(self)


class RecordCollection:
    def __init__(self):
        self.v1_meta: DatasetInfo = DatasetInfo()
        self.v2_meta: DatasetInfo = DatasetInfo()
        self.part_meta: DatasetInfo = DatasetInfo()
        self.partnet_meta: DatasetInfo = DatasetInfo()
        self.sem_meta: DatasetInfo = DatasetInfo()
        self.content: dict[str, Record] = {}

    def __len__(self):
        return len(self.content)

    def __str__(self):
        return obj_to_str(self)


def obj_to_str(obj):
    return json.dumps(obj, default=lambda o: o.__dict__)


def get_min_max_center(pcd):
    minimum = np.array([np.Infinity, np.Infinity, np.Infinity])
    maximum = np.array([-np.Infinity, -np.Infinity, -np.Infinity])
    avg = np.zeros(3)
    num_verts = 0
    for pnt in pcd:
        avg += pnt
        num_verts += 1
        minimum = np.minimum(pnt, minimum)
        maximum = np.maximum(pnt, maximum)
    center = avg / num_verts
    return maximum, minimum, center


def normalize_points(pcd):
    result = []
    maximum, minimum, center = get_min_max_center(pcd)
    diag = np.array(maximum) - np.array(minimum)
    norm = 1 / np.linalg.norm(diag)
    for v in pcd:
        v_new = (v - center) * norm
        result.append(v_new)
    return np.stack(result)


def read_obj(fn):
    # Reading .obj using Open3d would cause some errors with
    # object texture. Therefore, I used this manual approach
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
    vertices, faces = o3d.utility.Vector3dVector(
        vertices), o3d.utility.Vector3iVector(faces)
    return o3d.geometry.TriangleMesh(vertices, faces)


def get_icp_between_pointclouds(source, target, use_normal=False, threshold=1, trans_init=np.eye(4), max_iteration=2000):
    if use_normal:
        assert target.normals != None, "Target pointcloud must have normals."
    loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
    source_to_target = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return source_to_target.transformation
