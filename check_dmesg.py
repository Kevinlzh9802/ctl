import os
import time
import datetime


def check_shm_usage():
    process = os.popen('df -h')
    preprocessed = process.read()
    process.close()

    shm_info = preprocessed.split('\n')[1:-1]
    for ind in range(len(shm_info)):
        device_info_ind = [i for i in shm_info[ind].split(' ') if i != '']
        if device_info_ind[-1] == '/dev/shm':
            return int(device_info_ind[4][:-1])


def check_dmesg():
    process = os.popen('dmesg')
    preprocessed = process.read()
    process.close()
    return preprocessed


def while_loop(prev_flag, task_name):
    if prev_flag != 0:
        if check_shm_usage():
            return prev_flag, _, _
        else:
            dmesg = check_dmesg()
            return prev_flag + 1, dmesg, (datetime.datetime.now() - datetime.timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S')

    flag = True
    while flag:
        shm_num = check_shm_usage()
        with open(f'/datasets/imagenet100_results/{task_name}/checking_time.txt', 'a') as f:
            f.write(f"checking at {(datetime.datetime.now() - datetime.timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S')}\t shm={shm_num}%\n")
        print(f"checking at {(datetime.datetime.now() - datetime.timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S')}\t shm={shm_num}%")

        # checking groundtruth
        flag = (shm_num != 0)
        if not flag:
            dmesg = check_dmesg()
            return prev_flag + 1, dmesg, (datetime.datetime.now() - datetime.timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S')
        # checking period
        time.sleep(60)


task_name = 'ctl_rtc_imagenet100_trial3_DFS_seed500_retrain_from_task1'
while True:
    prev_flag = 0
    prev_flag, prev_demesg1, time_1 = while_loop(prev_flag, task_name)
    time.sleep(1200)
    prev_flag, prev_demesg2, time_2 = while_loop(prev_flag, task_name)
    if prev_flag == 1:
        continue
    time.sleep(1200)
    prev_flag, prev_demesg3, time_3 = while_loop(prev_flag, task_name)

    if prev_flag == 2:
        continue

    with open(f'/datasets/imagenet100_results/{task_name}1/dmesg.txt', 'a') as f:

        f.write(f'dmesg_1 at {time_1}\n')
        for i in prev_demesg1.split('\n')[-50:]:
            f.write(f'{i}\n')
        f.write(f'\n\ndmesg_2 at {time_2}\n')
        for i in prev_demesg2.split('\n')[-50:]:
            f.write(f'{i}\n')

        f.write(f'\n\ndmesg_3 at {time_3}\n')
        for i in prev_demesg3.split('\n')[-50:]:
            f.write(f'{i}\n')

    break

