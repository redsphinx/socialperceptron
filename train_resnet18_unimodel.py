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
# from deepimpression2.model_resnet18 import ResNet18


# settings
num_traits = 5
load_model = False


# device
if C.ON_GPU:
    _dev = 'cuda:%d' % C.DEVICE
else:
    _dev = 'cpu'
device = torch.device(_dev)


# get model
my_model = resnet18(pretrained=False)

my_model.fc = nn.Linear(in_features=512, out_features=num_traits, bias=True)
# my_model.fc = Tanh(nn.Linear(in_features=512, out_features=num_traits, bias=True))

if load_model:
    p = os.path.join(P.MODELS, '')
    my_model.load_state_dict(torch.load(p))
    print('model loaded')
    continuefrom = 0
else:
    continuefrom = 0

my_model.cuda(device)


# optimizer
learning_rate = 0.0002

my_optimizer = Adam(
            [
                {'params': my_model.conv1.parameters(), 'lr': learning_rate/64},
                {'params': my_model.bn1.parameters(), 'lr': learning_rate/32},
                {'params': my_model.layer1.parameters(), 'lr': learning_rate/16},
                {'params': my_model.layer2.parameters(), 'lr': learning_rate/8},
                {'params': my_model.layer3.parameters(), 'lr': learning_rate/4},
                {'params': my_model.layer4.parameters(), 'lr': learning_rate/2},
                {'params': my_model.fc.parameters(), 'lr': learning_rate}
            ],
            lr=learning_rate, betas=(0.5, 0.999), eps=10-8
        )

# loss
loss_function = L1Loss()

print('Initializing')
print('model initialized with %d parameters' % U.get_torch_params(my_model))

# train model
epochs = C.EPOCHS
# epochs = 1

train_labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
val_labels = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')
test_labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')

train_loss = []
pred_diff_train = np.zeros((epochs, num_traits), float)

val_loss = []
pred_diff_val = np.zeros((epochs, num_traits), float)

test_loss = []
pred_diff_test = np.zeros((epochs, num_traits), float)

training_steps = len(train_labels) // C.TRAIN_BATCH_SIZE
val_steps = len(val_labels) // C.VAL_BATCH_SIZE
test_steps = len(test_labels) // C.TEST_BATCH_SIZE
# training_steps = 2
# val_steps = 2

id_frames = h5.File(P.NUM_FRAMES, 'r')


def run(which, steps, which_labels, frames, model, optimizer, pred_diff, loss_saving, which_data, trait=None,
        ordered=False, save_all_results=False, record_predictions=False, record_loss=True, is_resnet18=True):
    print('steps: ', steps)
    assert(which in ['train', 'test', 'val'])
    assert(which_data in ['bg', 'face'])
    if trait is not None:
        assert(trait in ['O', 'C', 'E', 'A', 'S'])

    if which == 'train':
        which_batch_size = C.TRAIN_BATCH_SIZE
    elif which == 'val':
        which_batch_size = C.VAL_BATCH_SIZE
    elif which == 'test':
        which_batch_size = C.TEST_BATCH_SIZE

    loss_tmp = []
    pd_tmp = np.zeros((steps, num_traits), dtype=float)
    _labs = list(which_labels)

    preds = np.zeros((steps, num_traits), dtype=float)

    if not ordered:
        shuffle(_labs)

    ts = time.time()
    for s in tqdm(range(steps)):
        # HERE
        # if which == 'test':
        #     print(s)
        # HERE
        labels_selected = _labs[s * which_batch_size:(s + 1) * which_batch_size]
        assert (len(labels_selected) == which_batch_size)
        # labels, data, _ = D.load_data_single(labels_selected, which_labels, frames, which_data, resize=True,
        #                                   ordered=ordered, trait=trait)
        labels, data, _ = D.load_data(labels_selected, which_labels, frames, which_data, ordered=ordered,
                                      is_resnet18=is_resnet18)

        if C.ON_GPU:
            data = torch.from_numpy(data)
            data = data.cuda(device)
            labels = torch.from_numpy(labels)
            # labels = labels.long()
            labels = labels.cuda(device)

        if which == 'train':
            model.train()
            optimizer.zero_grad()
            predictions = model(data)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                predictions = model(data)
                loss = loss_function(predictions, labels)
                loss = loss.detach()

        if record_loss:
            loss_tmp.append(float(loss.data))
            # isn't this the same as MAE??
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

        U.record_loss_sanity(which, loss_tmp_mean, pred_diff[e])

        if which == 'test' and save_all_results:
            U.record_loss_all_test(loss_tmp, trait=True)

    if record_predictions and which == 'test':
        U.record_all_predictions(which, preds)


print('Enter training loop with validation')
for e in range(continuefrom, epochs):
    which_trait = None
    train_on = 'face'
    validate_on = 'face'
    test_on = 'face'
    # print('trained on: %s test on %s for trait %s' % (train_on, test_on, which_trait))
    # ----------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------
    run(which='train', steps=training_steps, which_labels=train_labels, frames=id_frames,
        model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_train,
        loss_saving=train_loss, which_data=train_on, trait=which_trait)
    # ----------------------------------------------------------------------------
    # validation
    # ----------------------------------------------------------------------------
    # C2.ORDERED_FRAME = 6 # TODO SET THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # run(which='val', steps=val_steps, which_labels=val_labels, frames=id_frames,
    #     model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_val,
    #     loss_saving=val_loss, which_data=validate_on, trait=which_trait, ordered=True)
    # ----------------------------------------------------------------------------
    # test
    # ----------------------------------------------------------------------------
    # C2.ORDERED_FRAME = 10 # TODO SET THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # times = 1
    # for i in range(1):
    #     if times == 1:
    #         ordered = True
    #         save_all_results = True
    #     else:
    #         ordered = False
    #         save_all_results = False
    # 
    #     run(which='test', steps=test_steps, which_labels=test_labels, frames=id_frames,
    #         model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_test,
    #         loss_saving=test_loss, which_data=test_on, ordered=ordered, save_all_results=save_all_results,
    #         trait=which_trait, record_loss=False, record_predictions=True)
    # best val 'bg': epoch_59_60_O, epoch_79_60_C, epoch_89_60_E, epoch_89_60_A, epoch_89_60_S
    # best val 'face' OCEAS: epoch_39_59_O, epoch_49_59_C, epoch_99_59_E, epoch_89_59_A, epoch_19_59_S

    # save model
    if ((e + 1) % 10) == 0:
        name = os.path.join(P.MODELS, 'epoch_%d_106' % e)
        torch.save(my_model.state_dict(), name)

# TODO: why is bg nan after epoch 4?? can't replicate...oh well