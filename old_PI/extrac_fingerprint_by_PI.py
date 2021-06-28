import os
import multiprocessing





def func(cluster_file, data_file, save_file):
    command='python PI_run.py {} {} {}'.format(cluster_file,data_file,save_file)
    os.system(command)

if __name__ == "__main__":

    data_dir = ''
    save_dir = ''
    label_num = 0
    dir_list = []
    param_lis = []

    for param_dir in dir_list:
        try:
            os.makedirs(os.path.join(save_dir, param_dir))
        except:
            pass
        for label in range(label_num):
            cluster_file = os.path.join(data_dir, param_dir, '{}_cluster'.format(label))
            data_file = os.path.join(data_dir, param_dir, '{}_hexdata'.format(label))
            save_file = os.path.join(save_dir, param_dir, '{}_result.txt'.format(label))
            param_lis.append([cluster_file, data_file, save_file])

    pool = multiprocessing.Pool(processes=20)
    for param in param_lis:
        [cluster_file, data_file, save_file]=param
        pool.apply_async(func, (cluster_file, data_file, save_file,))

    pool.close()
    pool.join()