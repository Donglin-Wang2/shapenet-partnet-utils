import os
import json
from typing import Optional, Tuple, Union
from configparser import ConfigParser, ExtendedInterpolation

import torch
import numpy as np
import open3d as o3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Pointclouds
from pytorch3d.loss import point_mesh_face_distance

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read('config.ini')
if not os.path.isdir(config['DEFAULT']['DATA_OUT_ROOT']):
    os.mkdir(config['DEFAULT']['DATA_OUT_ROOT'])
RECORD_PATH = config.get('DEFAULT', 'RECORD_PATH')


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
        self.registry: Optional[Registry] = None

    def __str__(self):
        return obj_to_str(self)


class DatasetInfo:
    def __init__(self):
        self.path: str = ''
        self.complete: bool = False
        self.registered: bool = False
        self.meta: Optional[object] = {}

    def __str__(self):
        return obj_to_str(self)


class Record:
    def __init__(self):
        self.id: str = ''
        self.acro_info: Optional[ItemInfo] = {}
        self.v1_info: Optional[ItemInfo] = {}
        self.v2_info: Optional[ItemInfo] = {}
        self.part_info: Optional[ItemInfo] = {}
        self.partnet_info: Optional[ItemInfo] = {}
        self.sem_info: Optional[ItemInfo] = {}

    def __str__(self):
        return obj_to_str(self)


class RecordCollection:
    def __init__(self):
        self.acro_meta: DatasetInfo = DatasetInfo()
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


def obj_to_str(obj: object) -> str:
    """Recursively convert objects to JSON strings given that all fields can be converted to strings.

    :param obj: Any object with other nested objects 
    :type obj: object
    :return: JSON string fo the object
    :rtype: str
    """
    return json.dumps(obj, default=lambda x: x.__dict__)


def point_to_mesh_dist(points: np.ndarray, mesh_path: str) -> float:
    """Calculate the distance between a point cloud in the form of a numpy ndarray and mesh in the form of a path. This uses the pytorch3d's point_mesh_face_distance() method. 

    :param points: (N, 3) numpy array container N points
    :type points: np.ndarray
    :param mesh_path: a file path for the mesh
    :type mesh_path: str
    :return: average unsigned distance between all points and the mesh
    :rtype: float
    """
    points = Pointclouds(torch.tensor(points).float().unsqueeze(0))
    meshes = load_objs_as_meshes([mesh_path])
    loss = point_mesh_face_distance(meshes, points)
    return loss.item()


def get_min_max_center(pcd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def get_registered_pointcloud(points: np.ndarray, registry: Registry, return_np: bool = False) -> Union[o3d.geometry.PointCloud, np.ndarray]:
    points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if registry.align is not None:
        points = points.rotate(registry.align)
    if registry.norm is not None:
        points = points.transform(registry.norm)
    if registry.reg is not None:
        points = points.transform(registry.reg)
    return np.array(points.points) if return_np else points


def get_normalize_matrix(pcd: np.ndarray, minimum: np.ndarray = None, maximum: np.ndarray = None, center: np.ndarray = None) -> np.ndarray:
    if minimum == None or maximum == None or center == None:
        maximum, minimum, center = get_min_max_center(pcd)
    diag = np.array(maximum) - np.array(minimum)
    norm = 1 / np.linalg.norm(diag)
    trans = np.eye(4)
    trans[:3, :3] *= norm
    trans[:3, 3] = -center
    return trans



def read_obj(fn: str):
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


def get_mesh_from_verices_and_faces(vertices: np.ndarray, faces: np.ndarray) -> o3d.geometry.TriangleMesh:
    vertices, faces = o3d.utility.Vector3dVector(
        vertices), o3d.utility.Vector3iVector(faces)
    return o3d.geometry.TriangleMesh(vertices, faces)


def get_icp_between_pointclouds(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, use_normal: bool = False, threshold: float = 1, trans_init: np.ndarray = np.eye(4), max_iteration: int = 2000):
    if use_normal:
        assert target.normals != None, "Target pointcloud must have normals."
    loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
    source_to_target = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    return source_to_target.transformation


def so3_to_se3(so3: np.ndarray) -> np.ndarray:
    se3 = np.concatenate(
        [so3, np.zeros((so3.shape[0], 1), dtype=np.float32)], axis=1)
    se3 = np.concatenate(
        [se3, np.array([0, 0, 0, 1], dtype=np.float32, ndmin=2)]
    )
    return se3