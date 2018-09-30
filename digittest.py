from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import cv2
import csv


def get_data():
    image_data = []
    filelist = []
    filepath = "./list/metertest_726_480x_aug_newloss_focal_se_epoch14_20180926.txt"
    with open(filepath) as f:
        for line in f.readlines():
            line_temp = line.split(' ')
            filename = line_temp[0]
            # print(filename)
            img = cv2.imread('/home/sk49/workspace/dataset/meter/' + filename)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # image = Image.open(image_path + filename)
            # image_data.append(np.array(, dtype='uint8'))
            for ann in range(1, len(line_temp)):
                # print(line_temp[ann])
                label_temp = line_temp[ann].split(',')
                # print(label_temp)
                # print(int(label_temp[0]),int(label_temp[1]),int(label_temp[2]),int(label_temp[3]))
                x = max(int(label_temp[0]), 0)
                y = max(int(label_temp[1]), 0)
                ex = int(label_temp[2]) + int(label_temp[0])
                ey = int(label_temp[1]) + int(label_temp[3])
                # print(x,y,ex,ey)
                img_crop = img[y:ey, x:ex, :]
                # print(img_crop.shape)

                img_crop = cv2.resize(img_crop, dsize=(64, 64))
                # cv2.imshow('imgcrop', img_crop)
                # print(img_crop.shape)
                # cv2.waitKey(0)
                image_data.append(np.array(img_crop, dtype='float32') / 255)
                # number = int(label_temp[4])
                # if number > 0:
                #     number = number
                # else:
                #     number = -number+11
                # label.append(number)
                filelist.append(filename)

    # np.savez(, image_data=image_data, box_data=box_data)
    # label = to_categorical(label)
    image_data = np.array(image_data)
    return image_data, filelist


def digit_predict(output_path, model_path):
    x, filelist = get_data()
    list_file = open(output_path, 'w')
    model = load_model(model_path)
    predict = model.predict(x)
    result = []
    filetemp = "test_mechanicalmeter20180921/5C0000.jpg"
    for i, line in enumerate(predict):
        index = np.argmax(line)
        label = index if index < 12 else -1
        if filelist[i] == filetemp:
            result.append(label)
        if i == len(predict) - 1 or filelist[i] != filetemp:
            labellist = ','.join([str(a) for a in result])
            list_file.write(filetemp + ' ' + labellist + '\n')
            filetemp = filelist[i]
            result.clear()
            result.append(label)

    list_file.close()
    print("predict Done!")


def evaluation(predict_path, truth_path):
    correct = 0
    truth = []
    result = []
    with open(truth_path, 'r') as fin:
        truthreader = csv.reader(fin)
        for row in truthreader:
            truth.append([row])

    with open(predict_path, 'r') as fin:
        resultreader = csv.reader(fin)
        for row in resultreader:
            result.append([row])

    for i, l in enumerate(truth):
        if l == result[i]:
            correct = correct + 1
    # print(correct/len(truth))
    return correct / len(truth)


def main():
    resultlist = './list/meter_alldigit8_726_480x_aug_newloss_focal_se_epoch14_20180926.txt'
    model_path = './checkpoints/models/train_digit-se-nas.003-0.065-0.9822.h5'
    digit_predict(resultlist, model_path)
    # truthlist = './meter_test_truth.csv'
    # acc = evaluation(resultlist,truthlist)
    # print(acc)


if __name__ == '__main__':
    main()
