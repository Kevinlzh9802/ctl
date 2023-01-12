import pandas as pd
import numpy as np

target_csv_path = '/Users/chenyuzhao/Downloads/eval_after_decouple/_task_10-3.csv'
df_target_csv = pd.read_csv(target_csv_path)

class_name_list = list(df_target_csv['class_index'])[1:]
avg_acc_list = list(df_target_csv['avg_acc'])[1:]
class_count_list = list(df_target_csv['count'])[1:]
acc_sum_list = list(df_target_csv['acc_sum'])[1:]

corase_level_index = -1
for i in range(len(class_name_list)):
    if '-' in class_name_list[i]:
        corase_level_index = i


avg_finest_level_acc = np.round(np.mean(avg_acc_list[corase_level_index+1:]), 3)
avg_total_acc = np.round(np.sum(acc_sum_list)/np.sum(class_count_list), 3)
if corase_level_index >= 0:
    avg_corase_level_acc = np.round(np.mean(avg_acc_list[:corase_level_index+1]), 3)
    print(f'avg_corase_level_acc:{avg_corase_level_acc} with {len(avg_acc_list[:corase_level_index+1])} classes, \navg_finest_level_acc: {avg_finest_level_acc} with {len(avg_acc_list[corase_level_index+1:])} classes, \navg_total_acc: {avg_total_acc} with {len(avg_acc_list)} classes')
else:
    print(f'avg_finest_level_acc: {avg_finest_level_acc} with {len(avg_acc_list[corase_level_index+1:])} classes, \navg_total_acc: {avg_total_acc} with {len(avg_acc_list)} classes')



