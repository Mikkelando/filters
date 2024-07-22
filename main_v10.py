from tqdm import tqdm
from utils.kalman_filer import stream_klmn_filter, get_filters
import numpy as np
import csv
import os

def load_landmarks(csv_path, qnt_l=468):

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data_all = [row for row in reader]
    x_list = []
    y_list = []
    for row_index,row in enumerate(data_all[1:]):
        # frame_num = float(row[0])
        # if int(frame_num)!= row_index+1:
        #     return None
        x_list.append([float(x) for x in row[0:0+qnt_l]])
        y_list.append([float(y) for y in row[0+qnt_l:0+qnt_l + qnt_l]])
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    landmark_array = np.stack([x_array,y_array],2)
    return landmark_array
def write_to_csv( data, file_path='data/lnd.csv', headers=None, qnt=468):
   
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            headers = [f'x_{i}' for i in range(qnt)] + [f'y_{i}' for i in range(qnt)]
 
            writer.writerow(headers)

        writer.writerow(data)




LND = np.array(load_landmarks('data/joe_face_lnd.csv'))
filters = get_filters(LND)
NEW_LND = []
NEW_LND.append(LND[0])
for i, lnd in tqdm(enumerate(LND[1:])):
    if i == 0:
        data = stream_klmn_filter(lnd, filters, LND[0] )
        NEW_LND.append(np.array([ [data[j][0][0][0][0], data[j][1][0][0][0]] for j in range(len(LND[0])) ]))
    else:
        data = stream_klmn_filter(lnd, filters, [ [data[i][0][0], data[i][1][0]] for i in range(len(LND[0])) ],
                                    prev_state_covariance= [ [data[i][0][1], data[i][1][1]] for i in range(len(LND[0])) ])
        
        # print(i, np.array([ [data[j][0][0][0][0], data[j][1][0][0][0]] for j in range(len(LND[0])) ]).shape)
        NEW_LND.append(np.array([ [data[j][0][0][0][0], data[j][1][0][0][0]] for j in range(len(LND[0])) ]))
        

NEW_LND = np.array(NEW_LND)


filters_rev = get_filters(LND)
NEW_LND_rev = []
NEW_LND_rev.append(LND[-1])
for i, lnd in tqdm(enumerate(LND[-2::-1])):
    if i == 0:
        data = stream_klmn_filter(lnd, filters_rev, LND[-1] )
        NEW_LND_rev.append(np.array([ [data[j][0][0][0][0], data[j][1][0][0][0]] for j in range(len(LND[0])) ]))
    else:
        data = stream_klmn_filter(lnd, filters_rev, [ [data[i][0][0], data[i][1][0]] for i in range(len(LND[0])) ],
                                    prev_state_covariance= [ [data[i][0][1], data[i][1][1]] for i in range(len(LND[0])) ])
        
        # print(i, np.array([ [data[j][0][0][0][0], data[j][1][0][0][0]] for j in range(len(LND[0])) ]).shape)
        NEW_LND_rev.append(np.array([ [data[j][0][0][0][0], data[j][1][0][0][0]] for j in range(len(LND[0])) ]))
        

NEW_LND = np.array(NEW_LND)
NEW_LND_rev = np.array(NEW_LND_rev)
print('REV: ', NEW_LND_rev.shape)

write_to_csv(NEW_LND, 'data/joe_tmp_new.csv')
write_to_csv(NEW_LND_rev, 'data/joe_tmp_new_rev.csv')

from os import listdir
from os.path import isfile, join
import cv2
onlyfiles = ['data/joe_face/'+ f for f in listdir('data/joe_face') if isfile(join('data/joe_face', f))]
print(onlyfiles[0])
fr = cv2.imread(onlyfiles[0])
height, width, layers = fr.shape
size = (width, height)


out1 = cv2.VideoWriter('data/joe_face_orig.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
out2 = cv2.VideoWriter(f'data/joe_face_stream.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
print('DRAWING')
for i, (file, lnd, new_lnd, new_lnd_rev) in tqdm(enumerate(zip(onlyfiles, LND, NEW_LND, NEW_LND_rev[::-1]))) :
    src = cv2.imread(file)
    f1 = src.copy()
    # bad_xs_indx =  CLST[i, :, 0]
    # bad_ys_indx =  CLST[i, :, 1]
    for j, (x, y) in enumerate(lnd):
        cv2.circle(f1, (int(x), int(y)), 2, (0, 255, 0), -1)
    for j, ((x, y) , (x_rev, y_rev)) in enumerate(zip(new_lnd, new_lnd_rev)):
        cv2.circle(src, (int((1.4*x + 0.6*x_rev)/2), int((1.4 * y + 0.6 * y_rev)/2)), 2, (0, 255, 0), -1)
        

    out1.write(f1)
    out2.write(src)

out1.release()
out2.release()
