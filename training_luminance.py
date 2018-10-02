# training on chalearn  mean image luminance with a simple perceptron
import chainer
from chainer.optimizers import Adam
from chainer.functions import mean_absolute_error
import numpy as np
from deepimpression2.model_64 import SimpleAll, SimpleOne
import deepimpression2.paths as P
import os
import deepimpression2.constants as C
import h5py as h5
from random import shuffle
from time import time
import deepimpression2.chalearn30.data_utils as D
from chainer.backends.cuda import to_gpu, to_cpu
import cupy as cp
import deepimpression2.util as U


all_traits = True

if all_traits:
    my_model = SimpleAll()
else:
    my_model = SimpleOne()

load_model = False
if load_model:
    p = os.path.join(P.MODELS, '')
    chainer.serializers.load_npz(p, my_model)
    print('model loaded')

my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
my_optimizer.setup(my_model)

if C.ON_GPU:
    my_model = my_model.to_gpu(device=C.DEVICE)

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

id_frames = h5.File(P.NUM_FRAMES, 'r')


def run(epoch, which, steps, which_labels, frames, model, optimizer, pred_diff, loss_saving, trait=None, ordered=False,
        save_all_results=False):
    print('steps: ', steps)
    assert(which in ['train', 'test', 'val'])
    if trait is not None:
        assert(trait in ['O', 'C', 'E', 'A', 'S'])

    if which == 'train':
        which_batch_size = C.TRAIN_BATCH_SIZE
    elif which == 'val':
        which_batch_size = C.VAL_BATCH_SIZE
    elif which == 'test':
        which_batch_size = C.TEST_BATCH_SIZE

    loss_tmp = []
    if trait is not None:
        pd_tmp = np.zeros((steps, 1), dtype=float)
    else:
        pd_tmp = np.zeros((steps, 5), dtype=float)
    _labs = list(which_labels)
    if not ordered:
        shuffle(_labs)

    ts = time()
    for s in range(steps):
        # HERE
        if which == 'test':
            print(s)
        # HERE
        labels_selected = _labs[s * which_batch_size:(s + 1) * which_batch_size]
        assert (len(labels_selected) == which_batch_size)

        labels, data, _ = D.load_data_luminance(labels_selected, which_labels, frames, trait, ordered)

        if C.ON_GPU:
            data = to_gpu(data, device=C.DEVICE)
            labels = to_gpu(labels, device=C.DEVICE)

        with cp.cuda.Device(C.DEVICE):
            if which == 'train':
                config = True
            else:
                config = False

            with chainer.using_config('train', config):
                if which == 'train':
                    model.cleargrads()
                prediction = model(data)

                loss = mean_absolute_error(prediction, labels)

                if which == 'train':
                    loss.backward()
                    optimizer.update()

        loss_tmp.append(float(loss.data))

        pd_tmp[s] = U.pred_diff_trait(to_cpu(prediction.data), to_cpu(labels))

    pred_diff[epoch] = np.mean(pd_tmp, axis=0)
    loss_tmp_mean = np.mean(loss_tmp, axis=0)
    loss_saving.append(loss_tmp_mean)
    print('E %d. %s loss: ' %(epoch, which), loss_tmp_mean,
          ' pred diff %s: ' % trait, pred_diff[epoch],
          ' time: ', time() - ts)

    U.record_loss_sanity(which, loss_tmp_mean, pred_diff[epoch])

    if which == 'test' and save_all_results:
        U.record_loss_all_test(loss_tmp, trait=True)


print('Enter training loop with validation')
for e in range(epochs):
    which_trait = 'O'  # [O C E A S]  or  None
    if which_trait is None:
        print('trained on luminance, for all 5 traits')
    else:
        print('trained on luminance, for trait %s' % which_trait)
    # print('trained on: %s test on %s for trait %s' % (train_on, test_on, which_trait))
    # ----------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------
    run(epoch=e, which='train', steps=training_steps, which_labels=train_labels, frames=id_frames, model=my_model,
        optimizer=my_optimizer, pred_diff=pred_diff_train, loss_saving=train_loss, trait=which_trait)
    # ----------------------------------------------------------------------------
    # validation
    # ----------------------------------------------------------------------------
    run(epoch=e, which='val', steps=val_steps, which_labels=val_labels, frames=id_frames, model=my_model,
        optimizer=my_optimizer, pred_diff=pred_diff_val, loss_saving=val_loss, trait=which_trait)
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
    #     run(epoch=e, which='test', steps=test_steps, which_labels=test_labels, frames=id_frames,
    #         model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_test,
    #         loss_saving=test_loss, ordered=ordered, save_all_results=save_results, trait=which_trait)
        # ordered=True so will not shuffle

    # best val 5 traits:
    # best val single traits OCEAS:

    # save model
    if ((e + 1) % 10) == 0:
        if which_trait is None:
            name = os.path.join(P.MODELS, 'epoch_%d_64' % (e))
        else:
            name = os.path.join(P.MODELS, 'epoch_%d_64_%s' % (e, which_trait))
        chainer.serializers.save_npz(name, my_model)
