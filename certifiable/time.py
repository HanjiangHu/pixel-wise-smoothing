from time import time
import datetime

predict_list = [
    "./data/predict/cms_new/metaroom/resnet101/tz/tv_noise_0.1_0.5" ,
    "./data/predict/save_all_new/resnet101/tz/tv_noise_0.01_0.5_7000_10k",
    "./data/predict/cms_new/metaroom/resnet101/ry/tv_noise_2.5_0.5",
    "./data/predict/save_all_new/resnet101/ry/tv_noise_0.25_0.5_7000_10k",
"./data/predict/cms_new/metaroom/resnet101/tx/tv_noise_0.05_0.5",
"./data/predict/save_all_new/resnet101/tx/tv_noise_0.005_0.5_7000_10k",
"./data/predict/cms_new/metaroom/resnet101/ty/tv_noise_0.05_0.5",
"./data/predict/save_all_new/resnet101/ty/tv_noise_0.005_0.5_7000_10k",
"./data/predict/cms_new/metaroom/resnet101/rz/tv_noise_2.5_0.5",
"./data/predict/save_all_new/resnet101/rz/tv_noise_0.7_0.5_7000_10k",
"./data/predict/cms_new/metaroom/resnet101/rx/tv_noise_2.5_0.5",
"./data/predict/save_all_new/resnet101/rx/tv_noise_0.25_0.5_7000_10k",
]
predict_list_75 = [
    "./data/predict/cms_new/metaroom/resnet101/tz/tv_noise_0.1_0.75",
    "./data/predict/save_all_new/resnet101/tz/tv_noise_0.01_0.75_7000_10k",
    "./data/predict/cms_new/metaroom/resnet101/ry/tv_noise_2.5_0.75",
    "./data/predict/save_all_new/resnet101/ry/tv_noise_0.25_0.75_7000_10k",
"./data/predict/cms_new/metaroom/resnet101/tx/tv_noise_0.05_0.75",
"./data/predict/save_all_new/resnet101/tx/tv_noise_0.005_0.75_7000_10k",
"./data/predict/cms_new/metaroom/resnet101/ty/tv_noise_0.05_0.75",
"./data/predict/save_all_new/resnet101/ty/tv_noise_0.005_0.75_7000_10k",
"./data/predict/cms_new/metaroom/resnet101/rz/tv_noise_2.5_0.75",
"./data/predict/save_all_new/resnet101/rz/tv_noise_0.7_0.75_7000_10k",
"./data/predict/cms_new/metaroom/resnet101/rx/tv_noise_2.5_0.75",
"./data/predict/save_all_new/resnet101/rx/tv_noise_0.25_0.75_7000_10k",
]
for path in predict_list:
    f2 = open(path,"r")
    lines = f2.readlines()
    total_time = datetime.datetime.strptime("00:00:00.0000", '%H:%M:%S.%f')
    sum = 0
    for i, line3 in enumerate(lines):
        if i == 0: continue
        if i > 120: break
        # print(line3[:-1].split(('\t'))[-1])
        time = datetime.datetime.strptime((line3[:-1].split(('\t'))[-1]), '%H:%M:%S.%f')
        delta = datetime.timedelta(hours=time.hour, minutes=time.minute, seconds=time.second, microseconds=time.microsecond)
        # time.
        total_time = total_time + delta
        sum += 1
        # print(i, total_time)
    total_second = total_time.hour * 3600 + total_time.minute * 60 + total_time.second
    # if path.split("/")[2] == "cms_new":
    #     total_second *= 7
    print(path, total_second/sum)