
import os
import time
from Config_bc import *

search_list = []

list_ce = [1]
print(list_ce)

list_mmd1 = [2, 5, 10, 100,
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
            0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            ]
list_mmd = list_mmd1[::-1]
# list_mmd = [0.0005, 0.008, 0.007, 0.006]
print(list_mmd)

f_write = f'param:\nLce:{list_ce}\nL-mmd:{list_mmd}'
f_name = f'./log_meta/{source_name}{target_name}-meta.txt'
print(f_write, f_name)
with open(f_name, "a+") as f:
    f.write(f_write + '\n')  # 自带文件关闭功能，不需要再写f.close()

for i in range(len(list_ce)):
    for j in range(len(list_mmd)):
        paramlist = [1, 1]
        paramlist[0] = list_ce[i]
        paramlist[1] = list_mmd[j]
        # print(paramlist)
        search_list.append(paramlist)
# print(search_list)



f_para_name = f'./log_meta/param-{source_name}{target_name}.txt'
num = 0
for param in search_list[num:]:
    print(param, num)
    root_latest = f"./weight/latest-meta-{num}-{source_name}{target_name}.pth"
    root_best = f"./weight/best-meta-{num}-{source_name}{target_name}.pth"
    paramlist = param
    with open(f_para_name, "a+") as f:
        f.truncate(0)
        f.write(root_latest + '\n')
        f.write(root_best + '\n')
        f.write(str(paramlist) + '\n')
    time.sleep(5)
    with open(f_name, 'a+') as f:
        f.write(str(num) + '\t')
    time.sleep(5)
    os.system('python3 meta_mmd_search_bc.py')
    num = num + 1




