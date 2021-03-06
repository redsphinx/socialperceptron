import skimage
from tqdm import tqdm
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
face_frame_30_test_split = np.zeros((len(unique_labels_names_test), 3, 208, 208))
# face_frame_30_test_split = np.zeros((100, 3, 208, 208))
face_frame_30_train_split = np.zeros((len(unique_labels_names_train), 3, 208, 208))

ff_frame_30_test_split = np.zeros((len(unique_labels_names_test), 3, 256, 456))
# ff_frame_30_test_split = np.zeros((100, 3, 256, 456))
ff_frame_30_train_split = np.zeros((len(unique_labels_names_train), 3, 256, 456))


def snr_per_image(image): # 3, h, w
    pixel_mean = image.mean()
    pixel_std = image.std()
    snr = pixel_mean / pixel_std
    if np.isnan(snr):
        return 0
    else:
        return snr


def entropy_per_image(image):
    entropy = skimage.measure.shannon_entropy(image)
    return entropy


def grab_20_different_faces():
    
    random_images = np.random.randint(len(unique_labels_names_train), size=20)
    for i in tqdm(random_images):
        
        name = unique_labels_names_train[i]
        h5_path = os.path.join(faces_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        frame_30 = Image.fromarray(np.transpose(frame_30.astype(np.uint8), (1, 2, 0)), mode='RGB')
        name = os.path.join(save_location_images, 'random_face_train_%d.jpg' % i)
        frame_30.save(name)


def grab_20_different_ff():
    random_images = np.random.randint(len(unique_labels_names_train), size=20)
    for i in tqdm(random_images):

        name = unique_labels_names_train[i]
        h5_path = os.path.join(entire_frame_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames - 1)][:]

        frame_30 = Image.fromarray(np.transpose(frame_30.astype(np.uint8)[0], (1, 2, 0)), mode='RGB')
        name = os.path.join(save_location_images, 'random_ff_train_%d.jpg' % i)
        frame_30.save(name)

# grab_20_different_ff()

# test
def snr_ff_test():
    for i in tqdm(range(len(unique_labels_names_test))):
    # for i in tqdm(range(100)):
        name = unique_labels_names_test[i]
        h5_path = os.path.join(entire_frame_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        ff_frame_30_test_split[i] = frame_30

    # avg over images: now we have the average face
    avg_face_test = ff_frame_30_test_split.mean(0)
    # std
    std_face_test_per_channel = ff_frame_30_test_split.std(0)
    # SNR = mean / std
    snr_face_test = avg_face_test.mean(0) / std_face_test_per_channel.mean(0)
    print('SNR ff test split: %f'  % (np.mean(snr_face_test)))
    snr_face_test /= np.max(snr_face_test)
    snr_face_test *= 100
    # save as image
    name = os.path.join(save_location_images, 'avg_ff_test.jpg')
    avg_face_test = np.transpose(avg_face_test, (1, 2, 0))
    avg_face_test = avg_face_test.astype(np.uint8)
    avg_face_test = Image.fromarray(avg_face_test, mode='RGB')
    avg_face_test.save(name)

    # collapse channels
    std_face_test_per_pixel = std_face_test_per_channel.mean(0)
    std_face_test_per_pixel = std_face_test_per_pixel.astype(np.uint8)
    std_face_test_per_pixel = Image.fromarray(std_face_test_per_pixel, mode='L')
    name = os.path.join(save_location_images, 'std_ff_test.jpg')
    std_face_test_per_pixel.save(name)

    snr_face_test = snr_face_test.astype(np.uint8)
    snr_face_test = Image.fromarray(snr_face_test, mode='L')
    name = os.path.join(save_location_images, 'snr_ff_test.jpg')
    snr_face_test.save(name)

# test
def snr_ff_test_2():
    div_by = len(unique_labels_names_test)
    # num = 100
    num = len(unique_labels_names_test)
    snr_total = 0
    for i in tqdm(range(num)):
        name = unique_labels_names_test[i]
        h5_path = os.path.join(entire_frame_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        # snr_image = snr_per_image(frame_30)
        snr_image = entropy_per_image(frame_30)
        if snr_image is not 0:
            snr_total += snr_image
        else:
            div_by -= 1

    snr_total /= div_by
    print('SNR Entire Frame Test: %s' % str(snr_total))

# train
def var_ff_train():
    num = len(unique_labels_names_train)
    for i in tqdm(range(num)):
        name = unique_labels_names_train[i]
        h5_path = os.path.join(entire_frame_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        ff_frame_30_train_split[i] = frame_30

    var = np.square(ff_frame_30_train_split.std(0))
    var = var.mean()
    print('var FF train: %s' % str(var))

# snr_ff_test_2()
# entropy Entire Frame Test: 7.271681837293523

def snr_ff_train_2():
    div_by = len(unique_labels_names_train)
    # num = 100
    num = len(unique_labels_names_train)
    snr_total = 0
    for i in tqdm(range(num)):
        name = unique_labels_names_train[i]
        h5_path = os.path.join(entire_frame_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        # snr_image = snr_per_image(frame_30)
        snr_image = entropy_per_image(frame_30)
        if snr_image is not 0:
            snr_total += snr_image
        else:
            div_by -= 1

    snr_total /= div_by
    print('SNR Entire Frame Train: %s' % str(snr_total))

# test
def var_ff_test():
    num = len(unique_labels_names_test)
    for i in tqdm(range(num)):
        name = unique_labels_names_test[i]
        h5_path = os.path.join(entire_frame_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        ff_frame_30_test_split[i] = frame_30

    var = np.square(ff_frame_30_test_split.std(0))
    var = var.mean()
    print('var FF test: %s' % str(var))

# snr_ff_test_2()
# entropy Entire Frame Test: 7.271681837293523
# snr_ff_train_2()
# entropy Entire Frame Train: 7.291229945121652


def snr_ff_train():
    for i in tqdm(range(len(unique_labels_names_train))):
    # for i in tqdm(range(100)):
        name = unique_labels_names_train[i]
        h5_path = os.path.join(entire_frame_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        ff_frame_30_train_split[i] = frame_30

    # avg over images: now we have the average face
    avg_face_train = ff_frame_30_train_split.mean(0)
    # std
    std_face_train_per_channel = ff_frame_30_train_split.std(0)
    # SNR = mean / std
    snr_face_train = avg_face_train.mean(0) / std_face_train_per_channel.mean(0)
    print('SNR ff train split: %f'  % (np.mean(snr_face_train)))
    snr_face_train /= np.max(snr_face_train)
    snr_face_train *= 100
    # save as image
    name = os.path.join(save_location_images, 'avg_ff_train.jpg')
    avg_face_train = np.transpose(avg_face_train, (1, 2, 0))
    avg_face_train = avg_face_train.astype(np.uint8)
    avg_face_train = Image.fromarray(avg_face_train, mode='RGB')
    avg_face_train.save(name)

    # collapse channels
    std_face_train_per_pixel = std_face_train_per_channel.mean(0)
    std_face_train_per_pixel = std_face_train_per_pixel.astype(np.uint8)
    std_face_train_per_pixel = Image.fromarray(std_face_train_per_pixel, mode='L')
    name = os.path.join(save_location_images, 'std_ff_train.jpg')
    std_face_train_per_pixel.save(name)

    snr_face_train = snr_face_train.astype(np.uint8)
    snr_face_train = Image.fromarray(snr_face_train, mode='L')
    name = os.path.join(save_location_images, 'snr_ff_train.jpg')
    snr_face_train.save(name)


# test
def snr_face_test_():
    for i in tqdm(range(len(unique_labels_names_test))):
    # for i in range(100):
        name = unique_labels_names_test[i]
        h5_path = os.path.join(faces_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        face_frame_30_test_split[i] = frame_30

    # avg over images: now we have the average face
    avg_face_test = face_frame_30_test_split.mean(0)  # (1600, 3, h, w)
    # std
    std_face_test_per_channel = face_frame_30_test_split.std(0)
    # SNR = mean / std
    snr_face_test = avg_face_test.mean(0) / std_face_test_per_channel.mean(0)
    print('SNR face test split: %f'  % (np.mean(snr_face_test)))
    snr_face_test /= np.max(snr_face_test)
    snr_face_test *= 100
    # save as image
    name = os.path.join(save_location_images, 'avg_face_test.jpg')
    avg_face_test = np.transpose(avg_face_test, (1, 2, 0))
    avg_face_test = avg_face_test.astype(np.uint8)
    avg_face_test = Image.fromarray(avg_face_test, mode='RGB')
    avg_face_test.save(name)

    # collapse channels
    std_face_test_per_pixel = std_face_test_per_channel.mean(0)
    std_face_test_per_pixel = std_face_test_per_pixel.astype(np.uint8)
    std_face_test_per_pixel = Image.fromarray(std_face_test_per_pixel, mode='L')
    name = os.path.join(save_location_images, 'std_face_test.jpg')
    std_face_test_per_pixel.save(name)

    snr_face_test = snr_face_test.astype(np.uint8)
    snr_face_test = Image.fromarray(snr_face_test, mode='L')
    name = os.path.join(save_location_images, 'snr_face_test.jpg')
    snr_face_test.save(name)

# test
def snr_face_test_2():
    div_by = len(unique_labels_names_test)
    # num = 100
    num = len(unique_labels_names_test)
    snr_total = 0
    for i in tqdm(range(num)):
        name = unique_labels_names_test[i]
        h5_path = os.path.join(faces_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        # snr_image = snr_per_image(frame_30)
        snr_image = entropy_per_image(frame_30)
        if snr_image is not 0:
            snr_total += snr_image
        else:
            div_by -= 1

    snr_total /= div_by
    print('SNR Faces Test: %s' % str(snr_total))

# test
def var_face_test():
    num = len(unique_labels_names_test)
    for i in tqdm(range(num)):
        name = unique_labels_names_test[i]
        h5_path = os.path.join(faces_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        face_frame_30_test_split[i] = frame_30

    var = np.square(face_frame_30_test_split.std(0))
    var = var.mean()
    print('var Faces Test: %s' % str(var))


# train
def snr_face_train_2():
    div_by = len(unique_labels_names_train)
    # num = 100
    num = len(unique_labels_names_train)
    snr_total = 0
    for i in tqdm(range(num)):
        name = unique_labels_names_train[i]
        h5_path = os.path.join(faces_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        # snr_image = snr_per_image(frame_30)
        snr_image = entropy_per_image(frame_30)
        if snr_image is not 0:
            snr_total += snr_image
        else:
            div_by -= 1

    snr_total /= div_by
    print('SNR Faces Train: %s' % str(snr_total))

# train
def var_face_train():
    num = len(unique_labels_names_train)
    for i in tqdm(range(num)):
        name = unique_labels_names_train[i]
        h5_path = os.path.join(faces_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        face_frame_30_train_split[i] = frame_30

    var = np.square(face_frame_30_train_split.std(0))
    var = var.mean()
    print('var Faces train: %s' % str(var))
    

# snr_ff_test_2()
# entropy Entire Frame Test: 7.271681837293523
# snr_ff_train_2()
# entropy Entire Frame Train: 7.291229945121652
# snr_face_test_2()
# entropy Faces Test: 7.370380135184912
# snr_face_train_2()
# entropy Faces Train: 7.4080031426104265

# train
def snr_face_train_():
    for i in tqdm(range(len(unique_labels_names_train))):
    # for i in range(100):
        name = unique_labels_names_train[i]
        h5_path = os.path.join(faces_location, name)
        h5_file = h5.File(h5_path, 'r')
        num_frames = len(h5_file.keys()) - 1
        if num_frames > 31:
            frame_30 = h5_file['30'][:]
        else:
            frame_30 = h5_file[str(num_frames-1)][:]

        face_frame_30_train_split[i] = frame_30

    # avg over images: now we have the average face
    avg_face_train = face_frame_30_train_split.mean(0)
    # std
    std_face_train_per_channel = face_frame_30_train_split.std(0)
    # SNR = mean / std
    snr_face_train = avg_face_train.mean(0) / std_face_train_per_channel.mean(0)
    print('SNR face train split: %f'  % (np.mean(snr_face_train)))
    snr_face_train /= np.max(snr_face_train)
    snr_face_train *= 100
    # save as image
    name = os.path.join(save_location_images, 'avg_face_train.jpg')
    avg_face_train = np.transpose(avg_face_train, (1, 2, 0))
    avg_face_train = avg_face_train.astype(np.uint8)
    avg_face_train = Image.fromarray(avg_face_train, mode='RGB')
    avg_face_train.save(name)

    # collapse channels
    std_face_train_per_pixel = std_face_train_per_channel.mean(0)
    std_face_train_per_pixel = std_face_train_per_pixel.astype(np.uint8)
    std_face_train_per_pixel = Image.fromarray(std_face_train_per_pixel, mode='L')
    name = os.path.join(save_location_images, 'std_face_train.jpg')
    std_face_train_per_pixel.save(name)

    snr_face_train = snr_face_train.astype(np.uint8)
    snr_face_train = Image.fromarray(snr_face_train, mode='L')
    name = os.path.join(save_location_images, 'snr_face_train.jpg')
    snr_face_train.save(name)




# snr_face_test_2()
# SNR Faces Test: 2.1899221201644634
# SNR Faces Test: 5.2634717272753715
# snr_face_train_2()
# SNR Faces Train: 2.2027206054816344
# snr_ff_test_2()
# SNR Entire Frame Test: 1.860420864901887
# SNR Entire Frame Test: 3.998158305489949
# snr_ff_train_2()
# SNR Entire Frame Train: 1.834831198354324

# var_face_test()
# var_face_train()
# var_ff_test()
var_ff_train()

# 100%|█████████████████████████████████████| 1676/1676 [00:01<00:00, 1101.15it/s]
# var Faces Test: 53.73953446432702
# 100%|█████████████████████████████████████| 6744/6744 [00:05<00:00, 1226.32it/s]
# var Faces train: 54.0868990233376
# 100%|██████████████████████████████████████| 1676/1676 [00:02<00:00, 768.23it/s]
# var FF test: 72.2594430153162
# 100%|██████████████████████████████████████| 6744/6744 [00:09<00:00, 705.89it/s]
# var FF train: 71.63847790853823