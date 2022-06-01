import os
import json
import pickle
from typing import Tuple
import torch
from tqdm import tqdm
import pytorch3d as torch3d

import numpy as np
from shapenet_v2 import ShapenetV2

from utils import *

DATA_ROOT = config['DEFAULT']['DATA_ROOT_PART']
if not os.path.isdir(config.get('DEFAULT', 'DATA_OUT_PART')):
    os.mkdir(config.get('DEFAULT', 'DATA_OUT_PART'))
DATA_OUT = config.get('DEFAULT', 'DATA_OUT_PART')


class ShapenetPart(ShapenetV2):
    def __init__(self) -> None:
        super().__init__()

    def gen_record(self) -> None:
        cat_name_to_id = {}
        cat_id_to_name = {}
        dataset_meta = DatasetInfo()
        dataset_meta.path = DATA_ROOT

        with open(os.path.join(DATA_ROOT, 'synsetoffset2category.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                name, id = line.split('\t')
                name, id = name.strip(), id.strip()
                cat_id_to_name[id] = name
                cat_name_to_id[name] = id

        dataset_meta.meta['cat_name_to_id'] = cat_name_to_id
        dataset_meta.meta['cat_id_to_name'] = cat_id_to_name
        self.records.part_meta = dataset_meta

        for cat_id in tqdm(os.listdir(DATA_ROOT), desc="Generating Shapenet Part records"):
            path = os.path.join(DATA_ROOT, cat_id, 'points')
            if not os.path.isdir(path) or cat_id == 'train_test_split':
                continue
            for fn in os.listdir(path):
                id = fn.split('.')[0]
                record = self.records.content.get(id, Record())
                record.id = id.split('.')[0]
                info = ItemInfo()
                info.cat_id = cat_id
                info.cat_name = cat_id_to_name[cat_id]
                info.path = os.path.join(path, fn)
                record.part_info = info
                self.records.content[record.id] = record
        self.records.part_meta.complete = True
        self.save_records()

    def register_data(self) -> None:
        if not self.records.v2_meta.complete:
            super().gen_record()
        for record in tqdm(self.records.content.values(), desc="Registering Shapenet Part records"):
            if not (record.part_info and record.v2_info):
                continue
            points = self.read_data(record.part_info.path)
            v2_verts, v2_faces = read_obj(record.v2_info.path)
            v2_mesh = get_mesh_from_verices_and_faces(v2_verts, v2_faces)
            v2_points = v2_mesh.sample_points_uniformly(
                8192, use_triangle_normal=True).normalize_normals()
            registry = Registry()
            registry.align = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            registry.norm = get_normalize_matrix(points)
            point_cloud = get_registered_pointcloud(points, registry)
            registry.reg = get_icp_between_pointclouds(point_cloud, v2_points)
            record.part_info.registry = registry

        self.records.part_meta.registered = True
        self.save_records()

    def calc_loss(self) -> None:
        count = 0
        for record in tqdm(self.records.content.values(), desc="Calculating Shapenet Part registration loss"):
            if not (record.part_info and record.v2_info):
                continue
            part_info = record.part_info
            assert part_info.registry
            registry = record.part_info.registry
            points = self.read_data(part_info.path)
            points = get_registered_pointcloud(
                points, registry, return_np=True)
            registry.loss += point_to_mesh_dist(
                points, record.content.v2_info.path)
            count += 1
        registry.loss = registry.loss / count

        self.save_records()

    def write_data(self, register: bool = False) -> None:

        if not self.records.part_meta.complete:
            print("Shapenet Part records have not been generated!")
            self.gen_record()
        if register and not self.records.v2_meta.complete:
            print("Shapenet Core v2 records have not been generated for registration!")
            super().gen_record()
        if register and not self.records.part_meta.registered:
            print("Shapenet Part records have not been registered!")
            self.register_data()
        
        points, point_labels, shape_labels = [], [], []
        for record in tqdm(self.records.content.values(), desc="Writing Shapenet Part records to disk"):
            if not record.part_info:
                continue
            if register:
                if not record.v2_info:
                    continue
                point = self.read_data(record.part_info.path)
                point = get_registered_pointcloud(
                    point, record.part_info.registry, return_np=True)
            else:
                point = self.read_data(record.part_info.path)

            point_label, shape_label = self.read_labels(record)
            points.append(point)
            point_labels.append(point_label)
            shape_labels.append(shape_label)

        points = np.array(points, dtype=object)
        point_labels = np.array(point_labels, dtype=object)
        shape_labels = np.array(shape_labels)

        np.save(os.path.join(DATA_OUT, 'points'), points)
        np.save(os.path.join(DATA_OUT, 'point_labels'), point_labels)
        np.save(os.path.join(DATA_OUT, 'shape_labels'), shape_labels)

    def read_data(self, path: str) -> np.ndarray:
        points = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                p = [float(token) for token in l.split(' ')]
                points.append(p)

        return np.stack(points)

    def read_labels(self, record: Record) -> Tuple[np.ndarray, object]:
        point_labels = []
        shape_label = record.part_info.cat_name
        point_path = record.part_info.path
        rel_path = os.path.join(point_path, '..', '..',
                                'points_label', record.id + '.seg')
        point_label_path = os.path.realpath(rel_path)
        with open(point_label_path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                point_labels.append(int(l) - 1)

        return np.array(point_labels), shape_label
