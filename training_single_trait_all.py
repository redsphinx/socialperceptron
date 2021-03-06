# Q: will the model be able to predict single traits at a time better than all 5 traits at the same time?
# train models on each trait
# only all, initialize network like the two freeze stream network


import chainer
import numpy as np
from deepimpression2.model_59 import Deepimpression
from deepimpression2.model_59 import LastLayers
import deepimpression2.constants as C
from chainer.functions import sigmoid_cross_entropy, mean_absolute_error, softmax_cross_entropy
from chainer.optimizers import Adam
import h5py as h5
import deepimpression2.paths as P
# import deepimpression2.chalearn20.data_utils as D
import deepimpression2.chalearn30.data_utils as D
import time
from chainer.backends.cuda import to_gpu, to_cpu
import deepimpression2.util as U
import os
import cupy as cp
from chainer.functions import expand_dims
from random import shuffle
from tqdm import tqdm


my_model = LastLayers()
load_model = False
if load_model:
    p = os.path.join(P.MODELS, '')
    chainer.serializers.load_npz(p, my_model)
    print('my_model loaded')
    continuefrom = 0
else:
    continuefrom = 0

bg_model = Deepimpression()
p = os.path.join(P.MODELS, 'epoch_89_60_A')
chainer.serializers.load_npz(p, bg_model)
print('bg model loaded')

face_model = Deepimpression()
p = os.path.join(P.MODELS, 'epoch_89_59_A')
chainer.serializers.load_npz(p, face_model)
print('face model loaded')

my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8, weight_decay_rate=0.001)
# my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
my_optimizer.setup(my_model)

if C.ON_GPU:
    my_model = my_model.to_gpu(device=C.DEVICE)
    bg_model = bg_model.to_gpu(device=C.DEVICE)
    face_model = face_model.to_gpu(device=C.DEVICE)

print('Initializing')
print('model initialized with %d parameters' % my_model.count_params())

epochs = C.EPOCHS
# epochs = 1

train_labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
val_labels = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')
test_labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')

train_loss = []
pred_diff_train = np.zeros((epochs, 1), float)

val_loss = []
pred_diff_val = np.zeros((epochs, 1), float)

test_loss = []
pred_diff_test = np.zeros((epochs, 1), float)

training_steps = len(train_labels) // C.TRAIN_BATCH_SIZE
val_steps = len(val_labels) // C.VAL_BATCH_SIZE
test_steps = len(test_labels) // C.TEST_BATCH_SIZE

# training_steps = 10
# val_steps = 10

id_frames = h5.File(P.NUM_FRAMES, 'r')


def run(which, steps, which_labels, frames, model, optimizer, pred_diff, loss_saving, trait, ordered=False,
        save_all_results=False, twostream=False, same_frame=False, record_loss=True, record_predictions=False):
    print('steps: ', steps)
    assert(which in ['train', 'test', 'val'])
    assert(trait in ['O', 'C', 'E', 'A', 'S'])

    if which == 'train':
        which_batch_size = C.TRAIN_BATCH_SIZE
    elif which == 'val':
        which_batch_size = C.VAL_BATCH_SIZE
    elif which == 'test':
        which_batch_size = C.TEST_BATCH_SIZE

    loss_tmp = []
    pd_tmp = np.zeros((steps, 1), dtype=float)
    _labs = list(which_labels)

    preds = np.zeros((steps, 1), dtype=float)

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

        labels_bg, bg_data, frame_num = D.load_data_single(labels_selected, which_labels, frames, which_data='bg',
                                                           resize=True, ordered=ordered, twostream=twostream,
                                                           same_frame=same_frame, trait=trait)
        labels_face, face_data, _ = D.load_data_single(labels_selected, which_labels, frames, which_data='face',
                                                       resize=True, ordered=ordered, twostream=twostream,
                                                       frame_num=frame_num, same_frame=same_frame, trait=trait)

        # shuffle data and labels in same order
        # TODO: this is issue...leave it. no wait, it's not. we repeat validation later without shuffling, saulgoodman
        if which != 'test':
            shuf = np.arange(which_batch_size)
            shuffle(shuf)
            bg_data = bg_data[shuf]
            face_data = face_data[shuf]
            labels_bg = labels_bg[shuf]

        if C.ON_GPU:
            bg_data = to_gpu(bg_data, device=C.DEVICE)
            face_data = to_gpu(face_data, device=C.DEVICE)
            labels = to_gpu(labels_bg, device=C.DEVICE)

        with cp.cuda.Device(C.DEVICE):
            if which == 'train':
                config = True
            else:
                config = False

            with chainer.using_config('train', False):
                prediction_bg, bg_activations = bg_model(bg_data)
                prediction_face, face_activations = face_model(face_data)

            with chainer.using_config('train', config):
                if config:
                    model.cleargrads()
                prediction = model(bg_activations, face_activations)

                loss = mean_absolute_error(prediction, labels)

                if which == 'train':
                    loss.backward()
                    optimizer.update()

        if record_loss:
            loss_tmp.append(float(loss.data))
            pd_tmp[s] = U.pred_diff_trait(to_cpu(prediction.data), to_cpu(labels))
        if record_predictions and which == 'test':
            preds[s] = to_cpu(prediction.data)

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
    which_trait = 'A'  # O C E A S
    train_on = 'all'
    validate_on = 'all'
    # print('trained on: %s val on: %s for trait %s' % (train_on, validate_on, which_trait))
    test_on = 'all'
    # print('trained on: %s val/test on %s for trait %s' % (train_on, test_on, which_trait))
    # ----------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------
    run(which='train', steps=training_steps, which_labels=train_labels, frames=id_frames, model=my_model,
        optimizer=my_optimizer, pred_diff=pred_diff_train, loss_saving=train_loss, trait=which_trait, same_frame=False)
    # ----------------------------------------------------------------------------
    # validation
    # ----------------------------------------------------------------------------
    # run(which='val', steps=val_steps, which_labels=val_labels, frames=id_frames, model=my_model,
    #     optimizer=my_optimizer, pred_diff=pred_diff_val, loss_saving=val_loss, trait=which_trait, same_frame=True)
    # ----------------------------------------------------------------------------
    # test
    # ----------------------------------------------------------------------------
    # times = 1
    # for i in range(1):
    #     if times == 1:
    #         ordered = True
    #         save_results = True
    #     else:
    #         ordered = False
    #         save_results = False
    #
    #     run(which='test', steps=test_steps, which_labels=test_labels, frames=id_frames,
    #         model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_test,
    #         loss_saving=test_loss, ordered=ordered, save_all_results=save_results, trait=which_trait,
    #         record_loss=True, record_predictions=True) # ordered=True so will not shuffle

    # save model
    if ((e + 1) % 10) == 0:
        name = os.path.join(P.MODELS, 'epoch_%d_127_%s' % (e, which_trait))
        chainer.serializers.save_npz(name, my_model)
'''
RESULTS

before:
best val 'all', no decay: epoch_79_61_O, epoch_89_61_C, epoch_69_61_E, epoch_29_61_A, epoch_29_61_S

best val 'bg': epoch_59_60_O, epoch_79_60_C, epoch_89_60_E, epoch_89_60_A, epoch_89_60_S
best val 'face': epoch_39_59_O, epoch_49_59_C, epoch_99_59_E, epoch_89_59_A, epoch_19_59_S

best val 'all', with decay 0.001: epoch_69_95_O, epoch_99_95_C, epoch_29_95_E, epoch_19_95_A, epoch_59_95_S

as of 1 apr 19:

best val 'all': 

best val 'bg': epoch_89_60_O, epoch_79_60_C, epoch_99_60_E, epoch_89_60_A, epoch_89_60_S
best val 'face': epoch_39_59_O, epoch_19_59_C, epoch_99_59_E, epoch_89_59_A, epoch_19_59_S



'''


