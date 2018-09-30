import csv
import random

with open("train_groundtruth_meter_-1_retinanet.csv", "r") as fr1:
    with open("train_groundtruth_meter_-1_retinanet_minusval.csv", "w", newline="") as fw1:
        with open("val_groundtruth_meter_-1_retinanet.csv", "w", newline="") as fw2:
            r1 = csv.reader(fr1)
            w1 = csv.writer(fw1)
            w2 = csv.writer(fw2)
            first = next(r1)
            w1.writerow(["imgname", "startx", "starty", "endx", "eny", "label_-1"])
            w2.writerow(["imgname", "startx", "starty", "endx", "eny", "label_-1"])
            a = list(r1)
            print(len(a))
            # print(a[0], type(a[0]))
            temp_name = a[0][0]
            b_temp = a[0][1:]
            data_dict = {}
            for l in a:
                if l[0] != temp_name:
                    data_dict[temp_name] = b_temp
                    print(b_temp)
                    b_temp = l[1:]
                    temp_name = l[0]
                else:
                    b_temp.extend(l[1:])
            keys = list(data_dict.keys())
            print(len(keys))
            print(keys)
            random.shuffle(keys)
            print(keys)
            lens = int(len(keys) * 0.9)

            for i, key in enumerate(keys):
                print("l", key)
                if i < lens:
                    value = list(data_dict[key])
                    for l in range(0, len(value) // 5):
                        temp = [key]
                        temp.extend(value[l * 5:l * 5 + 5])
                        print(temp)
                        w1.writerow(temp)
                else:
                    value = list(data_dict[key])
                    for l in range(0, len(value) // 5):
                        temp = [key]
                        temp.extend(value[l * 5:l * 5 + 5])
                        print(temp)
                        w2.writerow(temp)
