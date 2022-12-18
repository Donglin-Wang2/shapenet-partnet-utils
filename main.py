from shapenet_part import *

if __name__ == '__main__':
    shapenet_part = ShapenetPart()
    shapenet_part.gen_records()
    shapenet_part.register_data()
    shapenet_part.write_data(register=True)