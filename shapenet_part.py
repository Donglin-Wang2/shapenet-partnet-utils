import os
import json
import pickle
from tqdm import tqdm

import numpy as np

from utils import *

DATA_ROOT = config['DEFAULT']['DATA_ROOT_PART']
RECORD_PATH = config['DEFAULT']['RECORD_PATH']


class ShapenetPart():
    def __init__(self) -> None:
        self.records: RecordCollection = RecordCollection()

        if RECORD_PATH and os.path.isfile(RECORD_PATH):
            with open(RECORD_PATH, 'rb') as f:
                self.records = pickle.load(f)

    def get_records(self, use_json: bool = False) -> None:
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

        for cat_id in os.listdir(DATA_ROOT):
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

    def register_points(self) -> None:
        for id, record in tqdm(self.records.content.items()):
            if not (record.part_info and record.v2_info):
                continue
            pnts = self.read_points(record.part_info.path)
            v2_verts, v2_faces = read_obj(record.v2_info.path)
            v2_mesh = get_mesh_from_verices_and_faces(v2_verts, v2_faces)
            v2_points = v2_mesh.sample_points_uniformly(
                8192, use_triangle_normal=True).normalize_normals()
            registry = Registry()
            registry.align = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            pnts = pnts @ registry.align.T
            pnts = normalize_points(pnts)
            pnts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pnts))
            registry.reg = get_icp_between_pointclouds(pnts, v2_points)
            record.part_info.registry = registry
        
        self.save_records()
        
    def save_records(self):
        with open(RECORD_PATH, 'wb') as f:
            pickle.dump(self.records, f)

    def produce_data(self) -> None:
        pass
        
    def read_points(self, path: str) -> np.ndarray:
        pnts = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                p = [float(token) for token in l.split(' ')]
                pnts.append(p)
        return np.stack(p)
