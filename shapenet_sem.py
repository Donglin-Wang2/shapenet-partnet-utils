import csv
from utils import *
from shapenet_v2 import ShapenetV2

from tqdm import tqdm

DATA_ROOT = config['DEFAULT']['DATA_ROOT_SEM']
if not os.path.isdir(config.get('DEFAULT', 'DATA_OUT_SEM')):
    os.mkdir(config.get('DEFAULT', 'DATA_OUT_SEM'))
DATA_OUT = config.get('DEFAULT', 'DATA_OUT_SEM')

class ShapenetSem(ShapenetV2):
    def __init__(self):
        super().__init__()

    def gen_record(self) -> None:
        meta_path = os.path.join(DATA_ROOT, 'metadata.csv')
        sem_meta = DatasetInfo()
        with open(meta_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['fullId'][5:9] == 'room':
                    continue
                if not row['fullId'] or len(row['fullId']) < 5:
                    continue
                meta = {}
                if row['wnsynset']:
                    meta['cat'] = row['wnsynset'][1:]
                if row['up']:
                    meta['up'] = [float(ele) for ele in row['up'].split('\,')]
                if row['front']:
                    meta['front'] = [float(ele) for ele in row['front'].split('\,')]
                if row['aligned.dims']:
                    meta['aligned.dims'] = [
                        float(ele) for ele in row['aligned.dims'].split('\,')]
                if row['unit']:
                    meta['unit'] = float(row['unit'])
                
                sem_meta.meta[row['fullId'][5:]] = meta
                
        data_path = os.path.join(DATA_ROOT, 'models-OBJ', 'models')

        for fn in tqdm(os.listdir(data_path)):
            if fn.startswith('room'):
                continue
            id, ext = fn.split('.')
            if ext != 'obj':
                continue
            record = self.records.content.get(id, Record())
            record.id = id
            info = ItemInfo()
            if id in sem_meta.meta:
                info.cat_id = sem_meta.meta[id]['cat']
            info.path = os.path.join(data_path, fn)
            record.sem_info = info
            self.records.content[record.id] = record
        
        self.records.sem_meta.complete = True
        self.save_records()

    def register_data(self) -> None:
        if not self.records.v2_meta.complete:
            super().gen_record()
        
        self.records.sem_meta.registered = True
        self.save_records()