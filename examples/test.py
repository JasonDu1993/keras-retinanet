import cv2
import keras
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.preprocessing.open_images import get_labels

# import miscellaneous modules
import matplotlib.pyplot as plt

import os
import numpy as np
from time import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# use this environment flag to change which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


def bb(aa):
    return aa


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
        model_path = os.path.join('..', 'snapshots', '180817_resnet50_oid.epoch074-loss1.9542-valloss1.9309.h5')
    # model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
    model = models.load_model_custom(model_path, backbone_name='resnet50', convert=True)
    print(model.summary())
    return model


def predict_one_image(model, img_path):
    """predict the image with the model

    Args:
        model: A keras.models.Model object.
        img_path: the path of image

    Returns:
        draw: numpy.ndarray.including the box info
        boxes: List, the value is a list,length is 6. 6 represent x, y, w, h, label, score
    """
    starttime = time()
    image = read_image_bgr(img_path)
    import platform
    machine_name = platform.node()
    if machine_name == 'P100v0':
        metadata_dir = "/home/sk49/workspace/dataset/open_images_dataset_v4/challenge2018"
    else:
        metadata_dir = r'files/pictures/train_10'

    labels_to_names, cls_index = get_labels(
        metadata_dir=metadata_dir, version="challenge2018")
    print(labels_to_names)
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    cv2.imwrite("a.jpg", draw)
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time()

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("predicting time: ", time() - start)

    # correct for image scale
    boxes /= scale
    # visualize detections
    boxes = []
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        print(type(b), b)
        cv2.imwrite("jj.jpg", draw)
        draw_box(draw, b, color=color)
        print("2", type(draw))
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        print("3", type(draw))
        draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
        cv2.imwrite("jj1.jpg", draw)
        print("total", (time() - starttime), "s")

    return draw


def show_one_image(draw):
    plt.figure(figsize=(15, 15))
    plt.figure()
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


def predict():
    starttime = time()

    modelname = '180816'
    epoch = '57'
    n = 'all'  # 目前没有用
    mode = "test"  # val or test
    weight_path = os.path.join('files', 'checkpoints', 'models',
                               '1808016_alllayer_stage3.epoch057-loss17.99-valloss17.03.h5')
    model_path = os.path.join('files', 'checkpoints', 'models',
                              'bottleneck_oid.h5')

    #  when you test, only modify the above parameters
    yolo = YOLO(weight_path=weight_path, model_path=model_path)
    if mode == "val":
        test_csv_name = "challenge-2018-val-images-2000.csv"
    elif mode == "test":
        test_csv_name = "test_challenge_image.csv"
    else:
        raise Exception("mode only val or test")
    test_csv_path = os.path.join('files', mode + '_files', test_csv_name)
    pred_csv_path = os.path.join('files', mode + '_files',
                                 modelname + '_' + mode + '_' + epoch + 'epoch_' + strftime('%m%d%H%M') + '.csv')
    eval_csv_path = os.path.join('files', mode + '_files',
                                 modelname + '_' + mode + '_' + epoch + 'epoch_' + '2000_' + strftime(
                                     '%m%d%H%M') + '.csv')
    if mode == "val":
        dataset_dir = r"/home/sk49/workspace/dataset/open_images_dataset_v4/train"
    elif mode == "test":
        dataset_dir = r"/home/sk49/workspace/dataset/open_images_dataset_v4/test_challenge_2018"
    # dataset_dir = "files/pictures/train_10"

    print('dataset_dir', dataset_dir)

    cnt_total = 0
    pred_file = open(pred_csv_path, "w", newline="")
    with open(test_csv_path, 'r', newline='') as f:
        with open(eval_csv_path, "w", newline="") as writer_2000:
            reader = csv.reader(f)
            next(reader)
            writer = csv.writer(pred_file)
            writer_2000.write("ImageID,LabelName,Score,XMin,YMin,XMax,YMax\n")
            writer.writerow(["ImageId", "PredictionString"])
            for line in reader:
                cnt_total += 1
                img_id = line[0]
                img_name = img_id + ".jpg"
                print('now predicting %d image: %s' % (cnt_total, img_name))
                img_path = os.path.join(dataset_dir, img_name)
                if cnt_total % 2000 == 0:
                    print("%d images %.3fs" % (cnt_total, (time() - starttime)))
                r_image, boxes = yolo.detect_image(img_path)
                ih, iw, ic = r_image.shape
                # print('boxes', boxes)
                img_pred = []
                img_pred.append(img_id)
                s = ""
                if len(boxes) == 0:
                    writer_2000.write((img_id + ",/m/061hd_,1.0,0.0,0.0,1.0,1.0" + "\n"))
                else:
                    for x, y, w, h, label, score in boxes:
                        # if w > 0 and h > 0:
                        line = [img_id, str(label), str(score), str(x / iw), str(y / ih), str((x + w) / iw),
                                str((y + h) / ih)]
                        writer_2000.write(",".join(line) + "\n")
                        s += str(label) + " " + str(score) + " " + str(x / iw) + " " + str(y / ih) + " " + str(
                            (x + w) / iw) + " " + str((y + h) / ih) + " "
                # print('s', s)
                s = s[:-1]
                img_pred.append(s)
                writer.writerow(img_pred)
    print('total %f s' % (time() - starttime))
    yolo.close_session()
    pred_file.close()


if __name__ == '__main__':
    # predict()
    img_path = '4a3c61b76364514e.jpg'
    img_path = '000000008021.jpg'
    model = load_inference_model()
    predict_one_image(model, img_path)
