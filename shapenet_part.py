import os
import json
import pickle
from attr import has
from tqdm import tqdm

import numpy as np
from utils import DatasetInfo, ItemInfo, RecordCollection, normalize_points

from utils import Record, Registry, config, read_obj, get_icp_between_pointclouds

DATA_ROOT = config['DEFAULT']['DATA_ROOT_PART']
RECORD_PATH = config['DEFAULT']['RECORD_PATH']


class ShapenetPart():
    def __init__(self) -> None:
        self.records: RecordCollection = RecordCollection()

        # if RECORD_PATH and os.path.isfile(RECORD_PATH):
        #     with open(RECORD_PATH, 'rb') as f:
        #         self.records = pickle.load(f)
        #     if not hasattr(self.records, 'part_meta') or not self.records.part_meta.complete:
        #         self.get_records()
        # else:
        #     self.get_records()

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
            path = os.path.join(DATA_ROOT, cat_id)
            if not os.path.isdir(path) or cat_id == 'train_test_split':
                continue
            for id in os.listdir(path):
                record = Record()
                record.id = id.split('.')[0]
                info = ItemInfo()
                info.cat_id = cat_id
                info.cat_name = cat_id_to_name[cat_id]
                info.path = os.path.join(path, id)
                record.part_info = info
                self.records.content[record.id] = record

        self.records.part_meta.complete = True
        print("Here")
        if use_json:
            with open(RECORD_PATH, 'w') as f:
                json.dump(self.records, f)
        else:
            print(len(self.records))
            pickle.dump(self.records, open(RECORD_PATH, 'wb'))

    def register_points(self) -> np.ndarray:
        for id, record in tqdm(self.records.items()):
            if not (hasattr(record, 'part_inf') and hasattr(record, 'v2_info')):
                continue
            pnts = self.read_points(record.part_info.path)
            v2_mesh = read_obj(record.v2_info.path)
            v2_points = v2_mesh.sample_points_uniformly(
                8192, use_triangle_normal=True).normalize_normals()
            reg_mtx = Registry()
            reg_mtx.align = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            pnts = pnts @ reg_mtx.align.T
            reg_mtx.norm = normalize_points(pnts)
            reg_mtx.reg = get_icp_between_pointclouds(pnts, v2_points)

    def read_points(self, path: str) -> np.ndarray:
        pnts = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                p = [float(token) for token in l.split(' ')]
                pnts.append(p)
        return np.stack(p)