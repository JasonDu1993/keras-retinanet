import os
import numpy as np
import platform
from PIL import Image
from time import time
from yolo import YOLO
import csv

# from dataset_tool import DatasetTool

# Default anchor boxes
# YOLO_ANCHORS = np.array(((10, 13), (16, 30), (33, 23), (30, 61),
#                          (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)))
# YOLO_ANCHORS = np.array(((11,21),  (14,41),  (15,47),  (17,49),  (18,41),  (19,57),  (23,47),  (26,60),  (33,64)))

YOLO_ANCHORS = np.array(
    ((11, 21), (15, 41), (15, 47), (16, 47), (18, 41), (18, 53), (23, 47), (23, 60), (30, 62)))  # anchor2


# YOLO_ANCHORS = np.array(((13,24),  (17,47),  (17,54),  (19,54),  (21,47),  (21,61),  (26,54),  (26,70),  (35,72))) # anchor480
# YOLO_ANCHORS = np.array(((17,33),  (22,62),  (22,73),  (25,72),  (28,40),  (28,63),  (35,72),  (35,92),  (46,96))) # anchor640


def get_anchors(anchors_path):
    """loads the anchors from a file, split with ',', the two digtal represent the box the width, height
    for example 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326

    Args:
        anchors_path: the relative path of the anchors
    Returns:
        numpy.ndarray, shape (a,2), dtype=float64, the value is width, height, a represent the number of the anchors
    """
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
            return anchors
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS


# def draw_picture():
#     yolo = YOLO()
#     starttime = time()
#     dataset_dir = r'D:\DeepLearning\dataset\meter'
#     file_dir = os.path.dirname(os.path.abspath(__file__))
#     image_number = 0
#     with open(os.path.expanduser('keras_yolo3/list/meter_train_val.txt'), 'r') as f:
#         for line in f.readlines():
#             image_name = line.split(' ')[0].strip().split('/')
#             # print(image_name)
#             image_number = image_number + 1
#             img = Image.open(os.path.join(dataset_dir, image_name[0], image_name[1]))
#             print('image name:', image_name[1])
#             r_image = yolo.detect_image(img)
#             # r_image.show()
#             r_image.save(os.path.join(file_dir, 'picture', 'test_' + image_name[1]))
#     print('save %d images total %d s' % (image_number, (time() - starttime)))



def predict_test_image():
    # filePath = './model_data/test_groundtruth_index.txt'
    filePath = './model_data/test_label_gt_20180921.txt'
    # with open(filePath, 'r') as fin:
    #     reader = csv.reader(fin)
    #     img_list = list(reader)

    starttime = time()
    yolo = YOLO()
    machine_name = platform.node()
    if machine_name == 'P100v0':
        test_dir = '/home/sk49/workspace/dataset/meter/'
    elif machine_name == 'p100v1':
        test_dir = '/home/sk49/workspace/dataset/meter/'
    else:
        test_dir = './model_data/'
    # else:
    #     test_dir = input('please input the test image abspath dir:')
    # img_list = os.listdir(test_dir)
    # dataset = DatasetTool()
    list_file = open('./list/metertest_726_480x_aug_newloss_focal_se_epoch14_20180926.txt', 'w')
    image_number = 0
    with open(filePath, "r") as f:
        for row in f.readlines():
            filename = row.strip().split(' ')[0]
            # if imgName.startswith('test_') or imgName == 'predict':
            #     continue
            image_number = image_number + 1
            print('predict', image_number, 'image name:', filename)
            img = Image.open(os.path.join(test_dir, filename))
            # print('image name:', imgName)
            r_image, resultlist = yolo.detect_image(img)
            labels = []
            list_file.write(filename)
            for i, box in enumerate(resultlist):
                label = ",".join([str(a) for a in box[4].split(' ')])
                bbox = [box[0], box[1], box[2], box[3], label]
                bbox = ",".join([str(a) for a in bbox])
                list_file.write(" " + bbox)
                # labels.append(box[4].split(' ')[0])
            # labellist = ','.join([str(a) for a in labels])
            # list_file.write(filename + ' ' + labellist + '\n')
            # labellist = ",".join([str(a) for a in bbox])
            # list_file.write(filename + " " + box + "\n")
            list_file.write("\n")
            # dataset.save_annotation_table(int(row[0]), i, int(box[4].split(' ')[0]), box[0], box[1], box[2], box[3])
            save_path = "./list/726_480x_aug_newloss_focal_se_epoch14_20180926/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = 'predict_' + filename.split('/')[1]
            print(save_name)
            r_image.save(os.path.join(save_path, save_name))
    list_file.close()
    print('save %d images total %d s' % (image_number, (time() - starttime)))


if __name__ == '__main__':
    # draw_picture()
    predict_test_image()
