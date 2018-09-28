import platform


class Config(object):

    backbone = "resnet50"
    batch_size = 1
    gpu = 0
    multi_gpu = 0
    multi_gpu_force = True
    epochs = 150
    steps = 2000
    snapshot_path = "./snapshots"
    tensorboard_dir = "./logs"
    snapshots = False
    evaluation = True
    freeze_num = 0
    random_transform = True
    image_min_side = 800
    image_max_side = 1333
    machine_name = platform.node()
    version = "challenge2018"
    labels_filter = None  # type=csv_list
    annotation_cache_dir = "."
    parent_label = None

    if machine_name == 'P100v0':
        main_dir = '/home/sk49/workspace/dataset/open_images_dataset_v4'
    else:
        main_dir = r'E:\datasets\open_images_dataset_v4'

    dataset_type = "oid"

    @staticmethod
    def csv_list(string):
        return string.split(',')
