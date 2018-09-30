import cv2
import keras
import csv
import platform
import os
import sys
# import miscellaneous modules
import matplotlib.pyplot as plt

import os
import numpy as np
from time import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.preprocessing.open_images import get_labels
from keras_retinanet.preprocessing.csv_generator import _read_classes


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# use this environment flag to change which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


def load_inference_model(model_path=None):
    """ Load RetinaNet model
    adjust this to point to your downloaded/trained model
    models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
    if the model is not converted to an inference model, use the line below
    see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model

    Args:
        model_path: the path of model

    Returns:

    """
    if model_path is None:
        model_path = os.path.join('snapshots', '180929_resnet50.h5')
    # model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
    model = models.load_model_custom(model_path, backbone_name='resnet50', convert=True)
    model.load_weights("./snapshots/180929_resnet50_csv.epoch014-loss0.6687-valloss0.7717.h5")
    # print(model.summary())
    return model


def predict_one_image(model, img_path, score_iou=0.5):
    """predict the image with the model

    Args:
        model: A keras.models.Model object.
        img_path: the path of image

    Returns:
        draw: numpy.ndarray.including the box info
        pred: List, the value is a list,length is 6. 6 represent x1, y1, x2, y2, label, score
    """
    starttime = time()
    image = read_image_bgr(img_path)
    # cls_index, dict, the key is the class name, the value is the class index
    with open("./datas/class_mapping_meter_-1.csv", "r") as f:
        r = csv.reader(f)
        cls_index = _read_classes(r)

    labels_to_names = {}  # labels_to_names, dict, the key is the class_index, the value is the class name
    for key, value in cls_index.items():
        labels_to_names[value] = key
    # print("labels_to_names", labels_to_names)
    # copy to draw on
    draw = image.copy()
    # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("a.jpg", draw)
    # cv2.imshow("img", draw)
    # cv2.waitKey(0)
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time()

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print(img_path, "predicting time: ", time() - start)

    # correct for image scale
    boxes /= scale
    # visualize detections
    pred = []

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < score_iou:
            break
        pred_temp = []
        color = label_color(label)
        pred_temp.extend(box)
        pred_temp.append(labels_to_names[label])
        pred_temp.append(score)
        print("pred_temp", pred_temp)
        pred.append(pred_temp)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        # print("caption", caption, box)
        draw_caption(draw, b, caption, color=color)
        # draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("jj1.jpg", draw)
    pred.sort(key=lambda x: x[0])
    print(img_path, "total", (time() - starttime), "s")
    return draw, pred


def show_one_image(draw):
    plt.figure(figsize=(15, 15))
    plt.figure()
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


def predict_test_image():
    # filePath = './model_data/test_groundtruth_index.txt'
    gt_file_path = './model_data/test_label_gt_20180921.txt'
    # with open(gt_file_path, 'r') as fin:
    #     reader = csv.reader(fin)
    #     img_list = list(reader)

    starttime = time()
    model = load_inference_model()
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
    with open(gt_file_path, "r") as f:
        for row in f.readlines():
            filename = row.strip().split(' ')[0]
            # if imgName.startswith('test_') or imgName == 'predict':
            #     continue
            image_number = image_number + 1
            print('predict', image_number, 'image name:', filename)
            img = cv2.imwrite(os.path.join(test_dir, filename))
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
    # predict()
    img_path = '67.jpg'
    # img_path = '000000008021.jpg'
    model = load_inference_model()
    while True:
        img = input('Input image filename:')
        # dir_path = r'D:\DeepLearning\dataset\meter\mechanicalmeter'
        dataset_path = r'D:\DeepLearning\dataset\meter\test'
        img_path = os.path.join(dataset_path, img)
        try:
            draw, _ = predict_one_image(model, img_path)
            cv2.imshow("img1", draw)
            cv2.waitKey(0)
        except Exception as e:
            print('Open Error! Try again!', e)
            continue
