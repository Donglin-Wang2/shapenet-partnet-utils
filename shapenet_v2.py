
import os
import json
import pickle

from tqdm import tqdm

from utils import *

DATA_ROOT = config['DEFAULT']['DATA_ROOT_V2']


class ShapenetV2:
    def __init__(self) -> None:
        self.records: RecordCollection = RecordCollection()

        if RECORD_PATH and os.path.isfile(RECORD_PATH):
            with open(RECORD_PATH, 'rb') as f:
                self.records = pickle.load(f)

    def gen_record(self, use_json: bool = False) -> None:
        # Generating ShapenetCore v2 metadata
        dataset_meta = DatasetInfo()
        dataset_meta.path = DATA_ROOT
        with open(os.path.join(DATA_ROOT, 'taxonomy.json'), 'r') as f:
            meta = {obj['synsetId']: obj for obj in json.load(f)}
            dataset_meta.meta = meta
        self.records.v2_meta = dataset_meta
        # Reading the ShapenetCore v2 data
        for cat_id in tqdm(os.listdir(DATA_ROOT), desc="Generating Shapenet Core v2 records"):
            if cat_id == 'taxonomy.json':
                continue
            cat_path = os.path.join(DATA_ROOT, cat_id)

            for id in os.listdir(cat_path):

                path = os.path.join(cat_path, id, 'models')
                obj_path = os.path.join(path, 'model_normalized.obj')
                meta_path = os.path.join(path, 'model_normalized.json')

                if not (os.path.isdir(path) and os.path.isfile(obj_path) and os.path.isfile(meta_path)):
                    continue

                record = self.records.content.get(id, Record())
                record.id = id
                info = ItemInfo()
                info.cat_id = cat_id
                info.cat_name = meta[cat_id]['name']
                info.path = obj_path
                with open(meta_path, 'r') as f:
                    info.meta = json.load(f)
                record.v2_info = info
                self.records.content[id] = record

        self.records.v2_meta.complete = True
        self.save_records()

    def save_records(self):
        with open(RECORD_PATH, 'wb') as f:
            pickle.dump(self.records, f)

    def read_data(self):
        pass

    def write_data(self):
        pass