import os
import datetime
import time



def check_yaml(pod_name: object) -> object:
    process = os.popen(f'kubectl get pod {pod_name} -o yaml')
    preprocessed = process.read()
    process.close()
    return preprocessed

def check_desc(pod_name):
    process = os.popen(f'kubectl describe pod {pod_name}')
    preprocessed = process.read()
    process.close()
    return preprocessed

def check_logs(pod_name):
    process = os.popen(f'kubectl logs {pod_name}')
    preprocessed = process.read()
    process.close()
    return preprocessed

def check_cpu_usage(pod_name):
    process = os.popen(f'kubectl top pod {pod_name}')
    preprocessed = process.read()
    process.close()
    return preprocessed


pod_name = 'ctl-imagenet-4cpu-1gpu-64mem-pvc-datasets2-5-db4cddccc-4kldh'
save_path = f'/Users/chenyuzhao/Downloads/checking_pods/{pod_name}'
task_name = 'ctl_rtc_imagenet100_trial3_BFS_seed1993_retrain_from_task21_4'


while True:

    curr_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    mesg = ''
    mesg += f'pod_name: {pod_name}\n'
    mesg += f'task_name: {task_name}\n'
    mesg += f'checking_time: {curr_time}\n'
    mesg += f'\n\n\ncheck_yaml\n'
    mesg += check_yaml(pod_name)
    mesg += '\n\n\ncheck_yaml\n'
    mesg += check_desc(pod_name)
    mesg += '\n\n\ncheck_yaml\n'
    mesg += check_logs(pod_name)
    mesg += '\n\n\ncheck_cpu_usage\n'
    mesg += check_cpu_usage(pod_name)

    with open(f'{save_path}/{curr_time}.txt', 'a') as f:
        f.write(mesg)

    print(f'finsh at {curr_time}')

    if not check_yaml(pod_name):
        break

    time.sleep(1200)
