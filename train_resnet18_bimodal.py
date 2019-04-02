import numpy as np
import os
import h5py as h5
from random import shuffle
import time
from tqdm import tqdm

from torchvision.models import resnet18
from torch.optim.adam import Adam
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, L1Loss
from torch.nn import Tanh

import deepimpression2.paths as P
import deepimpression2.constants as C
import deepimpression2.chalearn20.constants as C2
import deepimpression2.util as U
import deepimpression2.chalearn30.data_utils as D
from deepimpression2.model_resnet18 import ResNet18_LastLayers


num_traits = 5

# face model
face_model = resnet18()
face_model.fc = nn.Linear(in_features=512, out_features=num_traits, bias=True)
p = os.path.join(P.MODELS, 'epoch_99_101')
face_model.load_state_dict(torch.load(p))
# get activations: make fc sequential
face_model.fc = nn.Sequential()

print('face model epoch_99_101 loaded')

# bg model
bg_model = resnet18()
bg_model.fc = nn.Linear(in_features=512, out_features=num_traits, bias=True)
p = os.path.join(P.MODELS, 'epoch_99_102')
bg_model.load_state_dict(torch.load(p))
# get activations: make fc sequential
bg_model.fc = nn.Sequential()
print('bg model epoch_99_102 loaded')

# final model
my_model = ResNet18_LastLayers(num_traits)
print('final model loaded')

# device
if C.ON_GPU:
    _dev = 'cuda:%d' % C.DEVICE
else:
    _dev = 'cpu'
device = torch.device(_dev)

face_model.cuda(device)
bg_model.cuda(device)
my_model.cuda(device)

learning_rate = 0.0002
my_optimizer = Adam(my_model.parameters(), lr=learning_rate, betas=(0.5, 0.999), eps=10-8)
loss_function = L1Loss()

print('Initializing')
print('model initialized with %d parameters' % U.get_torch_params(my_model))

# train model
epochs = C.EPOCHS
# epochs = 1

train_labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
train_loss = []
pred_diff_train = np.zeros((epochs, num_traits), float)
# training_steps = len(train_labels) // C.TRAIN_BATCH_SIZE
training_steps = 1
id_frames = h5.File(P.NUM_FRAMES, 'r')


def run(which, steps, which_labels, frames, model, optimizer, pred_diff, loss_saving, which_data, trait=None,
        ordered=False, save_all_results=False, record_predictions=False, record_loss=True, is_resnet18=True):
    print('steps: ', steps)
    assert(which in ['train'])
    assert(which_data in ['all'])
    if trait is not None:
        assert(trait in ['O', 'C', 'E', 'A', 'S'])

    which_batch_size = C.TRAIN_BATCH_SIZE

    loss_tmp = []
    pd_tmp = np.zeros((steps, num_traits), dtype=float)
    _labs = list(which_labels)
    preds = np.zeros((steps, num_traits), dtype=float)

    if which == 'train' and not ordered:
        shuffle(_labs)

    ts = time.time()
    for s in tqdm(range(steps)):
        labels_selected = _labs[s * which_batch_size:(s + 1) * which_batch_size]
        assert (len(labels_selected) == which_batch_size)
        
        # TODO: check if data are same size!!!!!!!
        # if not, load data with resize=True

        # face_data.shape
        # (32, 3, 256, 256)
        # bg_data.shape
        # (32, 3, 256, 256)

        labels_bg, bg_data, frame_num = D.load_data(labels_selected, which_labels, frames, which_data='bg',
                                                    ordered=ordered, is_resnet18=is_resnet18, same_frame=True)
        labels_face, face_data, _ = D.load_data(labels_selected, which_labels, frames, which_data='face',
                                                ordered=ordered, is_resnet18=is_resnet18, same_frame=True,
                                                frame_num=frame_num)
        # TODO: check if labels are same

        if C.ON_GPU:
            bg_data = torch.from_numpy(bg_data)
            bg_data = bg_data.cuda(device)
            face_data = torch.from_numpy(face_data)
            face_data = face_data.cuda(device)
            labels = torch.from_numpy(labels_bg)
            labels = labels.cuda(device)

        bg_model.eval()
        face_model.eval()
        with torch.no_grad():
            bg_activations = bg_model(bg_data)
            face_activations = face_model(face_data)

        model.train()
        optimizer.zero_grad()
        predictions = model(bg_activations, face_activations)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

        if record_loss:
            loss_tmp.append(float(loss.data))
            pd_tmp[s] = U.pred_diff_trait(np.array(predictions.cpu().data), np.array(labels.cpu().data))
        if record_predictions and which == 'test':
            preds[s] = np.array(predictions.cpu().data)

    if record_loss:
        pred_diff[e] = np.mean(pd_tmp, axis=0)
        loss_tmp_mean = np.mean(loss_tmp, axis=0)
        loss_saving.append(loss_tmp_mean)
        print('E %d. %s loss: ' %(e, which), loss_tmp_mean,
              ' pred diff %s: ' % trait, pred_diff[e],
              ' time: ', time.time() - ts)

        # U.record_loss_sanity(which, loss_tmp_mean, pred_diff[e])

        if which == 'test' and save_all_results:
            U.record_loss_all_test(loss_tmp, trait=True)

    if record_predictions and which == 'test':
        U.record_all_predictions(which, preds)


print('Enter training loop with validation')
for e in range(0, epochs):
    which_trait = None
    train_on = 'all'
    run(which='train', steps=training_steps, which_labels=train_labels, frames=id_frames,
        model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_train,
        loss_saving=train_loss, which_data=train_on, trait=which_trait)

    # save model
    # if ((e + 1) % 10) == 0:
    #     name = os.path.join(P.MODELS, 'epoch_%d_113' % e)
    #     torch.save(my_model.state_dict(), name)
