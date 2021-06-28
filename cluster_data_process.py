import os


dir=''
save_data=''
each_cluster_maxnum=0
os.makedirs(save_data,exist_ok=True)
dir_list=[]

for param_dir in dir_list:
    os.makedirs(os.path.join(save_data,param_dir),exist_ok=True)
    file_list=[os.path.join(dir,param_dir,file) for file in os.listdir(os.path.join(dir,param_dir))]
    for label,file in enumerate(file_list):
        cluster_dict={}
        data_list=[]
        with open(file,'r') as f:
            for index,data in enumerate(f.readlines()):
                [cluster_label,payload]=data.strip().split('\t')
                if(cluster_label not in cluster_dict):
                    cluster_dict[cluster_label]=[]
                cluster_dict[cluster_label].append(index)
                data_list.append(payload)


        with open(os.path.join(save_data,param_dir,'{}_cluster'.format(label)),'w') as f:
            count=-1
            for key,value in sorted(cluster_dict.items(), key=lambda e:e[0], reverse=False):
                split_num=len(value)//each_cluster_maxnum+1
                for i in range(0, len(value), each_cluster_maxnum):
                    now_value=value[i:i+each_cluster_maxnum]
                    count+=1
                    f.write('The Cluster {}:'.format(count)+'\n')
                    for num in now_value:
                        f.write(str(num)+' ')
                    f.write('\n')


        with open(os.path.join(save_data,param_dir,'{}_hexdata'.format(label)), 'w') as f:
            for data in data_list:
                f.write(data+'\n')



