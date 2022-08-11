from shapenet_sem import ShapenetSem
from utils import *
from shapenet_v2 import ShapenetV2

DATA_ROOT = config['DEFAULT']['DATA_ROOT_ACRO']
if not os.path.isdir(config.get('DEFAULT', 'DATA_OUT_ACRO')):
    os.mkdir(config.get('DEFAULT', 'DATA_OUT_ACRO'))
DATA_OUT = config.get('DEFAULT', 'DATA_OUT_ACRO')

class Acronym(ShapenetSem):
    def __init__(self):
        super().__init__()

    def gen_record(self) -> None:
        if not self.records.sem_meta.complete:
            super().gen_record()

        data_path = os.path.join(DATA_ROOT, 'grasps')
        for fn in os.listdir(data_path):
            cat, id, _ = fn.split("_")
            record = self.records.content.get(id, Record())
            record.id = id
            info = ItemInfo()
            info.cat_name = cat
            info.path = os.path.join(data_path, fn)
            record.acro_info = info
            self.records.content[record.id] = record
        self.records.acro_meta.complete = True
        self.save_records()

    def register_data(self) -> None:
        if not self.records.sem_meta.complete:
            super().gen_record()

        self.records.sem_meta.registered = True
        self.save_records()

if __name__ == '__main__':
    acronym = Acronym()
    acronym.gen_record()