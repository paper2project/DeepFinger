import os
import sys
import time

start=time.time()

with open(clusters, 'r') as f:
    text = f.readlines()
    clusternum=int(len(text) / 2)
    print('clusters num:', clusternum)

with open(save_file,'w') as f:
    f.write('')

clusternumin=clusternum if limit==0 else limit
print('show this time clusternum',clusternumin)

for i in range(clusternumin):
    os.system("python main.py -a {} {} {} {} {}".format(save_file,clusters,file,i+1,th))

print(time.time()-start)
