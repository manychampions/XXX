import pandas as pd
import os

def dropLess(srcRoot, noFaceCsv, srcAnnotationCsv):    #以漫画头像文件夹0漏检的数目决定丢弃那个id的文件夹
    noface_dataframe = pd.read_csv(noFaceCsv, sep=",")
    src_dataframe = pd.read_csv(srcAnnotationCsv, sep=",")
    noface_list = []
    id2count_dict = {}
    dropIdList = []
    for idx in range(len(noface_dataframe)):
        tmp = noface_dataframe.iloc[idx].values
        filename = tmp[1]
        id = int(filename.split("/")[-1].split("\\")[0])
        imagename = filename.split("/")[-1].split("\\")[-1]
        noface_list.append(imagename)
        if id not in id2count_dict.keys():
            id2count_dict[id] = 1
        elif id in id2count_dict.keys():
            id2count_dict[id] += 1

    for dir in os.listdir(srcRoot):
        folder = os.path.join(srcRoot, dir, '0')
        image_nums = len(os.listdir(folder))
        if int(dir) in id2count_dict.keys():
            noface_nums = id2count_dict[int(dir)]
            loss_ratio = noface_nums/float(image_nums)  #loss_ratio > 0.25 drop id
            if loss_ratio > 0.25:
                dropIdList.append(dir)
        else:
            continue

    dropIndexList = []
    for index in range(len(src_dataframe)):
        id = src_dataframe.iloc[index].values[2]
        imagename = src_dataframe.iloc[index].values[0]
        if str(id) in dropIdList or imagename in noface_list:
            dropIndexList.append(index)
        else:
            continue

    new_dataframe = src_dataframe.drop(index=dropIndexList)
    import pdb; pdb.set_trace()
    return new_dataframe




if __name__ == "__main__":
    srcRoot = "D:/YeJQ/Face_Pytorch-master/ksyun/af2019-ksyun-training-20190416/images/"
    noFaceCsv = "./nofaceList_0416.csv"
    srcAnnotationCsv = "D:/YeJQ/Face_Pytorch-master/ksyun/af2019-ksyun-training-20190416/annotations.csv"
    new_dataframe = dropLess(srcRoot, noFaceCsv, srcAnnotationCsv)
    new_dataframe.to_csv("./new_annotations.csv", index=0)
