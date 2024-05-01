import os
import argparse
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm
''' 
New
tz resnet50: 0.01 and 0.1
0.01 0.5sigma: 0.225
0.1 0.5sigma: 0.142 #0.082644
0.01 0.25sigma: 0.198347
0.1 0.25sigma: 0.0
0.01 0.75sigma:0.140495
0.1 0.75sigma: 0.090909

diffusion:
tz 0.01 0.5sigma 0.392, 1k
tz 0.02 0.5sigma 0.300, 1k
ry 0.25 0.5sigma 0.108, 1k
ry 0.5 0.5sigma 0.042, 1k

1 3 5 7, tz 0.01 0.5sigma resnet50: 
1000, 0.183
2000 0.192
3000, 0.192 2/ 217
4000: 0.217 main 0.223
5000,   0.217 main 0.223
6000: 0.225 main 0.223
7000, 0.225 main 0.231

34567, tz 0.02 0.5sigma resnet50: 
1000 0.167
2000 0.175
3000, 0.183
4000, 0.192 main 0.192
5000,   0.192 main 0.192
6000, 0.192 main 0.192
7000, 0.200 main 0.192

tz resnet101
0.01 0.5sigma: 0.15
0.1 0.5sigma: 0.10833
0.01 0.25sigma: 0.175
0.1 0.25sigma: 0.0
0.01 0.75sigma:0.140495
0.1 0.75sigma: 0.0416666

resnet101, tz 
0.01 0.5sigma 1000: 0.133 t
0.01 0.5sigma 2000: 0.133
0.01 0.5sigma 3000: 0.133
0.01 0.5sigma 3500: 0.133
0.01 0.5sigma 4000: 0.133
0.01, 0.5sigma 5000:0.133
0.01 0.5sigma 6000: 0.142
0.01 0.5sigma 7000: 0.15
0.02, 0.5, 3500: 0.133
0.02, 0.5sigma 5000, 0.133
0.02, 0.5sigma 6000, 0.133



ry  resnet50
0.25 0.25sigma: 0.025
0.25 0.5sigma: 0.15
0.25 0.75sigma: 0.108333
0.25, 0.5sigma, 1000: 0.08333
0.25, 0.5sigma, 2000:  0.117
0.25, 0.5sigma, 3000: 0.141666
0.25, 0.5sigma, 4000: 0.142
0.25, 0.5sigma, 5000: 0.15
0.25, 0.5sigma, 6000: 0.15
0.25, 0.5sigma, 7000: 0.15

0.5, 0.5sigma, 34567
3000: 0.075
4000, 0.1
5000, 0.092
5500: 0.117
6000:0.108
6500: 0.117
7000: 0.133

ry  resnet101
0.25 0.25sigma: 0.025
0.25 0.5sigma: 0.125
0.25 0.75sigma: 0.1
0.5, 0.5sigma: 0.125

resnet 101, 
0.25, 0.5sigma, 1000: 0.092
0.25, 0.5sigma, 2000: 0.092
0.25, 0.5sigma, 3000: 0.125
0.25, 0.5sigma, 3500: 0.125
0.25, 0.5sigma, 4000: 0.125
0.25, 0.5sigma, 5000: 0.125
0.25, 0.5sigma, 6000: 0.125
0.5, 0.5sigma, 3500, 0.108
0.5, 0.5sigma 5000, 0.108
0.5, 0.5sigma 6000, 0.117

tx resnet50
0.005 0.25sigma: 0
0.005 0.5sigma: 0.1916666
0.005 0.75sigma: 0.116666

3k 0.005 00.5sgima resnet 50 0.192

tx resnet101
0.005 0.25sigma: 0
0.005 0.5sigma: 0.16666
0.005 0.75sigma: 0.1

ty resnet50
0.005 0.25sigma: 0.0583
0.005 0.5sigma: 0.1833
0.005 0.75sigma: 0.11666

3k 0.005 0.5sigma resnet50 0.183

ty resnet101
0.005 0.25sigma: 0.075
0.005 0.5sigma: 0.133
0.005 0.75sigma: 0.13333

rx resnet50
0.25 0.25sigma: 0.04166
0.25 0.5sigma: 0.191666 main 0.142
0.25 0.75sigma: 0.11666

rx 3k 0.25 0.5sigma resnet50 0.175
4k 0.183
5k0.192

rx resnet101
0.25 0.25sigma: 0.0583333
0.25 0.5sigma: 0.14166
0.25 0.75sigma: 0.11666

rz resnet50
0.7 0.25sigma: 0
0.7 0.5sigma: 0.133
0.7 0.75sigma: 0.09166

3k 0.7 0.5sigma 0.133

rz resnet101
0.7 0.25sigma: 0
0.7 0.5sigma: 0.1166
0.7 0.75sigma: 0.075

baseline:
same amount CMS：
python analyze.py data/predict/cms_same/metaroom/resnet50/tz/tv_noise_0.02_0.5 data/results/cms_same/resnet50/tz/tv_noise_0.02_0.5 --step 0.01
python analyze.py data/predict/cms_same/metaroom/resnet50/tz/tv_noise_0.003_0.5 data/results/cms_same/resnet50/tz/tv_noise_0.003_0.5 --step 0.01

python analyze.py data/predict/cms_same/metaroom/resnet50/tz/tv_noise_0.01_0.5_uniform data/results/cms_same/metaroom/resnet50/tz/tv_noise_0.01_0.5_uniform --step 0.01


0.01: 0.491, 0.02: 0.483, python analyze.py data/predict/cms_same/metaroom/resnet50/tz/tv_noise_0.01_0.5 data/results/cms_same/resnet50/tz/tv_noise_0.01_0.5 --step 0.01




original CMS, time longer, 
0.01: 0.491, 0.02: 0.475
python analyze.py ~/projective_transformation/certifiable/data/predict/cms/metaroom/resnet50/tz/tv_noise_0.01_0.5 ~/projective_transformation/certifiable/data/results/cms/metaroom/resnet50/tz/tv_noise_0.01_0.5 --step 0.01


'''
parser = argparse.ArgumentParser(description='Analyze the real performance from logs')
parser.add_argument("logfile", help="path of the certify.py output")
parser.add_argument("outfile", help="the output path of the report")
parser.add_argument("--budget", type=float, default=0.0,
                    help="for semantic certification, the pre-allocated space for semantic transformations")
parser.add_argument("--step", type=float, default=0.25, help="step size for l2 robustness")
args = parser.parse_args()

def change_alpha(data):
    new_data = data.copy()
    for i in range(len(data)):
        if int(data["correct"][i]):
            radius = data["radius"][i]
            NA_list = []
            for j in range(1001):
                try_radius = 0.05 * norm.ppf(proportion_confint(j, 1000, alpha=2 * 0.01, method="beta")[0])
                if abs(radius - float("{:.3}".format(try_radius))) < 0.000001:
                    NA_list.append(j)
                    print(i, j, radius, float("{:.3}".format(try_radius)))
            # assert len(NA_list) == 1, f"{i}{NA_list}"
            new_radius = 0.05 * norm.ppf(proportion_confint(NA_list[-1], 1000, alpha=2 * 0.001, method="beta")[0])
            new_data["radius"][i] = float("{:.3}".format(new_radius))
        # assert 1==2
            # print(data["radius"][i])
    return new_data


if __name__ == '__main__':
    df = pd.read_csv(args.logfile, delimiter="\t")
    print(f'Total: {len(df)} records')
    # print(df)
    # df = change_alpha(df)
    steps = list()
    nums = list()
    now_step = args.budget
    while True:
        cnt = (df["correct"] & (df["radius"] >= now_step)).sum()
        mean = (df["correct"] & (df["radius"] >= now_step)).mean()
        steps.append(now_step)
        nums.append(mean)
        now_step += args.step
        if cnt == 0:
            break
    steps = [str(s) for s in steps]
    nums = [str(s) for s in nums]
    output = "\t".join(steps) + "\n" + "\t".join(nums)
    print(output)
    print(args.outfile)
    print(f'Output to {args.outfile}')
    if not os.path.exists(args.outfile):
        os.makedirs(args.outfile)
    f = open(args.outfile + "/certification_results.txt", 'w')
    print(output, file=f)
    print(f'Clean acc: {df["correct"].sum()}/{len(df)} = {df["correct"].sum()/len(df)}')
    f.close()
