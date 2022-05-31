
import os
import json
import pickle

from utils import DatasetInfo, ItemInfo, RecordCollection, config, Record

DATA_ROOT = config['DEFAULT']['DATA_ROOT_V2']
RECORD_PATH = config['DEFAULT']['RECORD_PATH']


class ShapenetV2:
    def __init__(self) -> None:
        self.records: RecordCollection = RecordCollection()

    def gen_record(self, use_json: bool = False) -> None:
        # Generating ShapenetCore v2 metadata
        dataset_meta = DatasetInfo()
        dataset_meta.path = DATA_ROOT
        with open(os.path.join(DATA_ROOT, 'taxonomy.json'), 'r') as f:
            meta = {obj['synsetId']: obj for obj in json.load(f)}
            dataset_meta.meta = meta
        self.records.v2_meta = dataset_meta
        # Reading the ShapenetCore v2 data
        for cat_id in os.listdir(DATA_ROOT):
            if cat_id == 'taxonomy.json':
                continue
            cat_path = os.path.join(DATA_ROOT, cat_id)

            for id in os.listdir(cat_path):

                path = os.path.join(cat_path, id, 'models')
                obj_path = os.path.join(path, 'model_normalized.obj')
                meta_path = os.path.join(path, 'model_normalized.json')

                if not (os.path.isdir(path) and os.path.isfile(obj_path) and os.path.isfile(meta_path)):
                    continue

                record = Record()
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

        if use_json:
            with open(RECORD_PATH, 'w') as f:
                json.dump(self.records, f)
        else:
            print(len(self.records))
            pickle.dump(self.records, open(RECORD_PATH, 'wb'))
