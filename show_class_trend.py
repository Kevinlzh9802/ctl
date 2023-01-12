import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def get_full_acc_list(csv_file_path):
    acc_summary_dict = {}
    csv_name_list = os.listdir(csv_file_path)
    for i in range(len(csv_name_list)):
        df = pd.DataFrame({})
        for j in csv_name_list:

            if (f'_task_{i}' == j.split('-')[0] or f'_task_{i}.csv'==j) and '.csv' in j and '_aux' not in j:
                df = pd.read_csv(f'{csv_file_path}/{j}')
                break
        if not df.shape[0] == 0:
            task_i_class = list(df['class_index'])
            task_i_class.remove('epoch_0')
            task_i_class = [int(i) for i in task_i_class]
            task_i_acc = list(df['avg_acc'])[1:]
            for class_i_index in range(len(task_i_class)):
                class_i = task_i_class[class_i_index]
                if class_i in acc_summary_dict:
                    acc_summary_dict[class_i]['acc_list'].append(task_i_acc[class_i_index])
                else:
                    acc_summary_dict[class_i] = {'start_at_task': i, 'acc_list': [task_i_acc[class_i_index]]}
    return acc_summary_dict

def generate_sub_plot(acc_summary_list, label_name):
    plt.plot([acc_summary_list['start_at_task']+ i for i in range(len(acc_summary_list['acc_list']))], acc_summary_list['acc_list'], label=label_name)


def plot_acc_list(acc_summary_dict_all, class_to_check):
    generate_sub_plot(acc_summary_dict_all['DER'][class_to_check], f'DER_class_{class_to_check}')
    generate_sub_plot(acc_summary_dict_all['DFS'][class_to_check], f'DFS_class_{class_to_check}')
    generate_sub_plot(acc_summary_dict_all['BFS'][class_to_check], f'BFS_class_{class_to_check}')
    plt.xlim(0, 25)
    # plt.ylim(0, 1)
    plt.legend()
    plt.show()

def plot_sub_full_avg_trend(csv_file_path, file_name, type='finest', balance=False):
    acc_summary_dict = {}
    csv_name_list = os.listdir(csv_file_path)
    acc_list = []
    for i in range(len(csv_name_list)):
        df = pd.DataFrame({})
        for j in csv_name_list:

            if (f'_task_{i}' == j.split('-')[0] or f'_task_{i}.csv'==j) and '.csv' in j and '_aux' not in j:
                df = pd.read_csv(f'{csv_file_path}/{j}')
                break
        if not df.shape[0] == 0:
            df_target_csv = df
            class_name_list = list(df_target_csv['class_index'])[1:]
            avg_acc_list = list(df_target_csv['avg_acc'])[1:]

            corase_level_index = -1
            for i in range(len(class_name_list)):
                if '-' in class_name_list[i]:
                    corase_level_index = i

            if type == 'finest':
                avg_finest_level_acc = np.round(np.mean(avg_acc_list[corase_level_index + 1:]), 3)
                acc_list.append(avg_finest_level_acc)
            else:
                avg_corase_level_acc = np.round(np.mean(avg_acc_list[:corase_level_index + 1]), 3)
                acc_list.append(avg_corase_level_acc)

    if balance:
        if file_name == 'DFS':
            acc_list = acc_list[2:]
        if 'BFS' in file_name:
            acc_list = acc_list[5:]

    plt.plot(range(len(acc_list)), acc_list, label=file_name)

def plot_full_avg_trend(csv_file_dict, type='finest', balance=False):

    for file_name in csv_file_dict:
        csv_file_path = csv_file_dict[file_name]
        plot_sub_full_avg_trend(csv_file_path, file_name, type=type, balance=balance)
    plt.xlim(0, 25)
    plt.show()

DER_csv_file_path = '/Users/chenyuzhao/Downloads/imagenet100_trial3_DER_seed1993_with_imagenet100_config_reuslt/eval_after_decouple'
DFS_csv_file_path = '/Users/chenyuzhao/Downloads/imagenet100_trial3_DFS_seed1993_with_imagenet100_config_reuslt/eval_after_decouple'
BFS_csv_file_path = '/Users/chenyuzhao/Downloads/imagenet100_trial3_BFS_seed500_with_imagenet100_config_reuslt/eval_after_decouple'

acc_summary_dict_all = {'DER': get_full_acc_list(DER_csv_file_path), 'BFS': get_full_acc_list(BFS_csv_file_path), 'DFS': get_full_acc_list(DFS_csv_file_path)}
plot_acc_list(acc_summary_dict_all, 844)

csv_file_dict = {'DER':DER_csv_file_path, 'DFS':DFS_csv_file_path, 'BFS': BFS_csv_file_path}
plot_full_avg_trend(csv_file_dict, type='', balance=False)


