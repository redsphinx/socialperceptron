import os
import numpy as np
import h5py as h5
from PIL import Image

import deepimpression2.paths as P
import deepimpression2.chalearn30.data_utils as D


save_location_images = '/scratch/users/gabras/data/chalearn_snr'

faces_location = P.CHALEARN_ALL_DATA_20_2
entire_frame_location = P.CHALEARN30_ALL_DATA
test_split_location = P.CHALEARN_TEST_SPLIT
train_split_location = P.CHALEARN_TRAIN_SPLIT

all_names_extended = os.listdir(faces_location)
labels_names_test = np.genfromtxt(test_split_location, str)
labels_names_train = np.genfromtxt(train_split_location, str)

unique_labels_names_test = []
unique_labels_names_train = []

# get the unique video names: asjkdfg.001
for i in range(len(all_names_extended)):
    if all_names_extended[i].split('.')[0] in labels_names_test:
        unique_labels_names_test.append(all_names_extended[i])
    elif all_names_extended[i].split('.')[0] in labels_names_train:
        unique_labels_names_train.append(all_names_extended[i])

# load the 30th frame from each video
face_frame_30_test_split = np.array((len(unique_labels_names_test), 3, 256, 256))
face_frame_30_train_split = np.array((len(unique_labels_names_train), 3, 256, 256))

# test
for i in range(len(unique_labels_names_test)):
    name = labels_names_test[i]
    h5_path = os.path.join(faces_location, name)
    h5_file = h5.File(h5_path, 'r')
    num_frames = len(h5_file.keys()) - 1
    if num_frames > 31:
        frame_30 = h5_file['30'][:]
    else:
        frame_30 = h5_file[str(num_frames-1)][:]

    face_frame_30_test_split[i] = frame_30

# avg over images: now we have the average face
avg_face_test = face_frame_30_test_split.mean(0)
# save as image
name = os.path.join(save_location_images, 'avg_face_test.jpg')
avg_face_test = Image.fromarray(avg_face_test, mode='RGB')
avg_face_test.save(name)
# std
std_face_test_per_channel = face_frame_30_test_split.std(0)
# collapse channels
std_face_test_per_pixel = std_face_test_per_channel.mean(0)
# convert colors to interpretable
# std_face_test_per_pixel *= 255
std_face_test_per_pixel = Image.fromarray(std_face_test_per_pixel, mode='L')
name = os.path.join(save_location_images, 'std_face_test.jpg')
std_face_test_per_pixel.save(name)




# SNR = mean^2 / variance
# SNR = mean / std

