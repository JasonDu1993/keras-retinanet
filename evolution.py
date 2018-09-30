import csv
import os


def evoluate(filepath):
    # truthpath = "./model_data/test_threeclass_groundtruth_label.txt"
    # truthpath = "./model_data/test_groundtruth_label.txt"
    truthpath = "/model_data/test_label_gt_20180921.txt"
    # truthpath = "./list/meter_trainall_label619.txt"
    # predictpath = "./list/meter_testall_712_anchor640_newlossall_focal_afterremove.txt"
    # truthpath = './list/meter_error619_new.txt'
    # predictpath = './list/meter_error619afterremove_new2.txt'
    # predictpath = "./list/meter_trainall_predict619_new.txt"
    # errorlist = open('./list/meter_testerror620afterremove.txt', 'w')
    # save_path = "./list/testall619/"
    predictpath = filepath
    with open(truthpath) as f:
        truth = f.readlines()
    print(len(truth))
    total = len(truth)
    correct = 0
    # errorlist = []
    with open(predictpath) as f:
        for i, row in enumerate(f.readlines()):
            if row == truth[i]:
                correct = correct + 1
                # filename = truth[i].split(" ")[0]
                # delete
                # os.remove(save_path + 'predict_' + filename.split('/')[1])
                # move
                # imagePath = save_path + 'predict_' + filename.split('/')[1]
                # dest = save_path + 'correct/predict_' + filename.split('/')[1]
                # os.rename(imagePath, dest)
                # else:
                # if(row.split(" ")[0] != truth[i].split(" ")[0]):
                #     print(row)
                #     print(truth[i])
                #     break;
                # errorlist.append(truth[i])
                # errorlist.write(truth[i])
    acc = correct / total
    # errorlist.close()
    print(correct)
    print(acc)


def evoluate_int_only(filepath):
    # truthpath = "./list/meter_trainall_label619.txt"
    # predictpath = "./list/meter_trainall_predict619_new.txt"
    # truthpath = "./model_data/test_threeclass_groundtruth_label.txt"
    # truthpath = "./model_data/test_groundtruth_label.txt"
    truthpath = "./model_data/test_label_gt_20180921.txt"
    # predictpath = "./list/meter_testall_712_anchor640_newlossall_focal_afterremove.txt"
    predictpath = filepath
    with open(truthpath) as f:
        truth = f.readlines()
    print(len(truth))
    total = len(truth)
    correct = 0
    with open(predictpath) as f:
        for i, row in enumerate(f.readlines()):
            labels = row.split(' ')[1].strip()
            labellist = labels.split(',')
            for j, l in enumerate(labellist):
                if l == '':
                    break
                elif int(l) < 0 or int(l) == 11:
                    labellist = labellist[:j]
                    break
            truelabels = truth[i].split(' ')[1].strip('\n')
            truelabellist = truelabels.split(',')
            for j, l in enumerate(truelabellist):
                if l == '':
                    break
                elif int(l) < 0 or int(l) == 11:
                    truelabellist = truelabellist[:j]
                    break
            # if '11' in labellist:
            #     index11 = labellist.index('11')
            #     labellist = ','.join(str(l) for l in labellist[:index11])
            #     true11 = truelabellist.index('11')
            #     truelabellist = ','.join(str(l) for l in truelabellist[:true11])
            # save_path = "./list/0914_416_aug_binaryloss_se_epoch32/"
            save_path = "./list/726_480x_aug_newloss_focal_se_epoch14_20180926/"
            if labellist == truelabellist:
                correct = correct + 1
                filename = truth[i].split(" ")[0]
                # # delete
                # # os.remove(save_path + 'predict_' + filename.split('/')[1])
                # move
                imagePath = save_path + 'predict_' + filename.split('/')[1]
                correct_path = os.path.join(save_path, "correct")
                if not os.path.exists(correct_path):
                    os.makedirs(correct_path)
                dest = save_path + 'correct/predict_' + filename.split('/')[1]
                os.rename(imagePath, dest)
    print(correct)
    acc = correct / total
    print(acc)


def digitcmp(truth, predict):
    lenth = len(truth) if len(truth) <= len(predict) else len(predict)
    for i in range(lenth):
        if truth[i] == predict[i]:
            continue;
        else:
            return i;
    return lenth


def evoluate_int_by_image(filepath):
    # truthpath = "./model_data/test_threeclass_groundtruth_label.txt"
    # truthpath = "./model_data/test_groundtruth_label.txt"
    truthpath = "./model_data/test_label_gt_20180921.tx"
    # predictpath = "./list/meter_testall_712_anchor640_newlossall_focal_afterremove.txt"
    predictpath = filepath
    with open(truthpath) as f:
        truth = f.readlines()
    print(len(truth))
    total = len(truth)
    correct = 0
    acc = []
    with open(predictpath) as f:
        for i, row in enumerate(f.readlines()):
            labels = row.split(' ')[1].strip('\n')
            labellist = labels.split(',')
            for j, l in enumerate(labellist):
                if l == '':
                    break;
                elif (int(l) < 0 or int(l) == 11):
                    labellist = labellist[:j]
                    break;
            truelabels = truth[i].split(' ')[1].strip('\n')
            truelabellist = truelabels.split(',')
            for j, l in enumerate(truelabellist):
                if l == '':
                    break;
                elif (int(l) < 0 or int(l) == 11):
                    truelabellist = truelabellist[:j]
                    break;
            # if '11' in labellist:
            #     index11 = labellist.index('11')
            #     labellist = ','.join(str(l) for l in labellist[:index11])
            #     true11 = truelabellist.index('11')
            #     truelabellist = ','.join(str(l) for l in truelabellist[:true11])
            if (len(truelabellist) == 0):
                print(row)
            accbyimage = digitcmp(truelabellist, labellist) / len(truelabellist)
            acc.append(accbyimage)
        totalacc = sum(acc) / total

    print(totalacc)


if __name__ == '__main__':
    # filepath = "./list/meter_testall_predict726_anchor480aug_newloss_249_focal_se_014_box_416.txt"
    # output = "./list/meter_testall_726_anchor480aug_newloss249_focal_se_014_afterremove_416.txt"
    #
    # deleteBox(filepath,output)
    # filepath1 = "./list/meter_testall_predict726_anchor480aug_newloss_249_focal_se_014_box_480.txt"
    # filepath2 = "./list/meter_testall_predict726_anchor480aug_newloss_249_focal_se_014_box_416.txt"
    # output = "./list/meter_testall_726_anchor480_480_416_afterremove.txt"
    #
    # mergeBox(filepath1,filepath2,output)
    output = "./list/meter_alldigit8_726_480x_aug_newloss_focal_se_epoch14_20180926.txt"
    # output = "./list/meter_alldigit8_0912_416_epoch67.txt"
    # print("evoluate all:")
    # evoluate(output)
    print("evoluate int only:")
    evoluate_int_only(output)
    print("evoluate boxes:")
    # evoluate_int_by_image(output)
