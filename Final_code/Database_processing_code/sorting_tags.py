import csv
import numpy as np
import pickle

f = open('../database/annotations_final.txt', 'r')

reader = csv.reader(f, delimiter='\t')
tags = next(reader)
annotation_dict = {}

while True:
    try:
        values = next(reader)
        # values1 = next(reader, None)
        annotation_dict[values[0]] = {}# data is a dict. values[0] is the clip id, which is the key->pointing to a dict of all tags
        for tagnames, value in zip(tags[1:], values[1:]):
            annotation_dict[values[0]][tagnames] = value
    except StopIteration:
        print('end tag annotations file')
        break

stat_mat = np.zeros((1,len(tags)-2))
# print(stat_mat.shape)
i=1
for key in annotation_dict:
    temp = list(annotation_dict[key].values())
    temp = temp[:-1]
    temp = np.array([temp])
    stat_mat = np.concatenate((stat_mat,temp),axis=0)
    i+=1
    if i%1000 == 0:
        print(i)

stat_mat = stat_mat[1:,:]
stat_mat = stat_mat.astype(np.int)
# print(stat_mat.shape)
stats={}
sums = np.sum(stat_mat,axis=0)
for i,tag in enumerate(tags[1:-1]):
    stats[tag] = sums[i]

i=1
for key,value in stats.items():
    print(i,key,':',value)
    i+=1

i=1
sorted_stats = sorted(stats.items(), key=operator.itemgetter(1),reverse=True)
# print(type(sorted_stats))
for key,val in sorted_stats:
    print(i,key,val)
    i+=1

with open('sorted_tags.pickle', 'wb') as handle:
    pickle.dump(sorted_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
