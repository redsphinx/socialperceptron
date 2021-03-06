from sklearn.linear_model import LinearRegression
import numpy as np
import deepimpression2.paths as P
import os
import h5py as h5
import deepimpression2.chalearn30.data_utils as D
import matplotlib
matplotlib.use('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


train_labels = h5.File(P.CHALEARN_TRAIN_LABELS_20, 'r')
val_labels = h5.File(P.CHALEARN_VAL_LABELS_20, 'r')
test_labels = h5.File(P.CHALEARN_TEST_LABELS_20, 'r')
id_frames = h5.File(P.NUM_FRAMES, 'r')


def linear_regression_all():
    labels, data, _ = D.load_data_special(list(train_labels), train_labels, id_frames, use_color=True)
    reg = LinearRegression().fit(data, labels)
    reg.score(data, labels)
    print('reg score: ', reg.score(data, labels)) # reg score:  0.012926151576513799

    labels, data, _ = D.load_data_special(list(test_labels), test_labels, id_frames, ordered=True, use_color=True)
    prediction = reg.predict(data)
    loss = np.abs(prediction - labels)
    loss = np.mean(loss, axis=1)
    # loss = np.mean(loss)
    print('loss: ', np.mean(loss))

    num = P.TEST_LOG.split('_')[-1].split('.')[0]
    save_path = os.path.join(P.LOG_BASE, 'testall_%s.txt' % num)
    with open(save_path, 'a') as mf:
        for i in range(len(loss)):
            line = '%s\n' % str(loss[i])[0:6]
            mf.write(line)


# linear_regression_all()

def linear_regression_single():
    labels, data, _ = D.load_data_special(list(train_labels), train_labels, id_frames, use_color=True)
    labels_test, data_test, _ = D.load_data_special(list(test_labels), test_labels, id_frames, ordered=True,
                                                    use_color=True)

    traits = ['O', 'C', 'E', 'A', 'S']
    for i, t in enumerate(traits):
        print('for trait: %s' % t)
        label_trait = labels[:,  i]
        reg = LinearRegression().fit(data, label_trait)

        print('reg score: ', reg.score(data, label_trait))

        prediction = reg.predict(data_test)
        loss = np.abs(prediction - labels_test[:, i])
        print('loss: ', np.mean(loss))

        num = P.TEST_LOG.split('_')[-1].split('.')[0]
        save_path = os.path.join(P.LOG_BASE, 'testall_%s_%s.txt' % (num, t))
        with open(save_path, 'a') as mf:
            for i in range(len(loss)):
                line = '%s\n' % str(loss[i])[0:6]
                mf.write(line)

    # for trait: O
    # reg score:  0.010992353074178562
    # loss:  0.11836213
    # for trait: C
    # reg score:  0.02270418985996847
    # loss:  0.12730783
    # for trait: E
    # reg score:  0.014756351764524054
    # loss:  0.12690265
    # for trait: A
    # reg score:  0.006859205152472847
    # loss:  0.10820333
    # for trait: S
    # reg score:  0.010139626895114384
    # loss:  0.12715155


# linear_regression_single()


def make_trait_lum_plot(num='75'):
    labels_test, data_test, _ = D.load_data_special(list(test_labels), test_labels, id_frames, ordered=True,
                                                    use_color=True)
    print('data loaded')

    def get_color(clr_code):
        g = int(clr_code * 100)
        # rgb = np.array([255, g, 0])
        # rgb = np.expand_dims(rgb, axis=0)
        return g

    traits = ['O', 'C', 'E', 'A', 'S']
    for i, t in enumerate(traits):
        labels = labels_test[:, i]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = data_test[:, 0]
        y = data_test[:, 1]
        z = data_test[:, 2]
        col = np.zeros((len(labels)))
        for j in range(len(labels)):
            col[j] = get_color(labels[j])
        ax.scatter(x, y, z, c=col)

        # for l in range(len(labels)):
        #     x, y, z = data_test[l]
        #     ax.scatter(x, y, z, c=get_color(labels[l]))

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        for angle in range(0, 360, 10):
            ax.view_init(30, angle)
            plt.draw()

        # x = data_test
        # y = labels
        # plt.scatter(x, y, s=1, alpha=0.8)
        # plt.title('trait %s as a function of frame luminance' % t)
        # plt.xlabel('frame luminance ')
        # plt.ylabel('trait intensity')

            save_path = os.path.join(P.FIGURES, 'train_%s' % num)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig('%s/luminance_angle_%d_%s.png' % (save_path, angle, t))


# make_trait_lum_plot()


# gender + ethnicity stuff
def linear_regression_all_GE():
    all_labels = D.load_labels_only(list(train_labels), train_labels)
    labels = all_labels[:, :5]
    data = all_labels[:, -2:]
    reg = LinearRegression().fit(data, labels)
    reg.score(data, labels)
    print('reg score: ', reg.score(data, labels))

    all_labels = D.load_labels_only(list(test_labels), test_labels)
    labels = all_labels[:, :5]
    data = all_labels[:, -2:]
    prediction = reg.predict(data)
    loss = np.abs(prediction - labels)
    loss = np.mean(loss, axis=1)
    print('loss: ', np.mean(loss))

    num = P.TEST_LOG.split('_')[-1].split('.')[0]
    save_path = os.path.join(P.LOG_BASE, 'testall_%s.txt' % num)
    with open(save_path, 'a') as mf:
        for i in range(len(loss)):
            line = '%s\n' % str(loss[i])[0:6]
            mf.write(line)


# linear_regression_all_GE()


def linear_regression_single_GE():
    all_labels = D.load_labels_only(list(train_labels), train_labels)
    labels = all_labels[:, :5]
    data = all_labels[:, -2:]

    all_labels_test = D.load_labels_only(list(test_labels), test_labels)
    labels_test = all_labels_test[:, :5]
    data_test = all_labels_test[:, -2:]

    traits = ['O', 'C', 'E', 'A', 'S']
    for i, t in enumerate(traits):
        if t == 'A':
            print('uo')
        print('for trait: %s' % t)
        label_trait = labels[:,  i]
        reg = LinearRegression().fit(data, label_trait)

        print('reg score: ', reg.score(data, label_trait))

        prediction = reg.predict(data_test)
        loss = np.abs(prediction - labels_test[:, i])
        print('loss: ', np.mean(loss))

        num = P.TEST_LOG.split('_')[-1].split('.')[0]
        save_path = os.path.join(P.LOG_BASE, 'testall_%s_%s.txt' % (num, t))
        with open(save_path, 'a') as mf:
            for i in range(len(loss)):
                line = '%s\n' % str(loss[i])[0:6]
                mf.write(line)

        # for trait: O
        # reg score:  0.03225619822554082
        # loss:  0.11579283
        # for trait: C
        # reg score:  0.011184243463907118
        # loss:  0.12864001
        # for trait: E
        # reg score:  0.045332631739610885
        # loss:  0.123488486
        # uo
        # for trait: A
        # reg score:  0.004047642024675668
        # loss:  0.10773547
        # for trait: S
        # reg score:  0.004874068347403004
        # loss:  0.12656394

# linear_regression_single_GE()


def ml_lum_clr_geth_all():
    # get luminance and color
    labels_train, data_train, _ = D.load_data_special(list(train_labels), train_labels, id_frames, use_everything=True)
    all_labels = D.load_labels_only(list(train_labels), train_labels)
    labels = all_labels[:, :5]
    # get gender and ethnicity
    data = all_labels[:, -2:]
    data = np.concatenate((data_train, data), axis=1)  # luminance 1, color 3, gender 1, ethnicity 1
    reg = LinearRegression().fit(data, labels)
    reg.score(data, labels)
    print('reg score: ', reg.score(data, labels))  # 0.02931919237388262

    labels_test, data_test, _ = D.load_data_special(list(test_labels), test_labels, id_frames, ordered=True,
                                                      use_everything=True)
    all_labels = D.load_labels_only(list(test_labels), test_labels)
    labels = all_labels[:, :5]
    data = all_labels[:, -2:]
    data = np.concatenate((data_test, data), axis=1)  # luminance 1, color 3, gender 1, ethnicity 1
    prediction = reg.predict(data)
    loss = np.abs(prediction - labels)
    loss = np.mean(loss, axis=1)
    print('loss: ', np.mean(loss))

    num = P.TEST_LOG.split('_')[-1].split('.')[0]
    save_path = os.path.join(P.LOG_BASE, 'testall_%s.txt' % num)
    with open(save_path, 'a') as mf:
        for i in range(len(loss)):
            line = '%s\n' % str(loss[i])[0:6]
            mf.write(line)


# ml_lum_clr_geth_all()


def ml_lum_clr_geth_single():
    labels_train, data_train, _ = D.load_data_special(list(train_labels), train_labels, id_frames, use_everything=True)
    all_labels = D.load_labels_only(list(train_labels), train_labels)
    labels = all_labels[:, :5]
    # get gender and ethnicity
    data = all_labels[:, -2:]
    data = np.concatenate((data_train, data), axis=1)  # luminance 1, color 3, gender 1, ethnicity 1

    labels_test, data_test, _ = D.load_data_special(list(test_labels), test_labels, id_frames, ordered=True,
                                                    use_everything=True)
    all_labels = D.load_labels_only(list(test_labels), test_labels)
    labels_test = all_labels[:, :5]
    data_geth = all_labels[:, -2:]
    data_test = np.concatenate((data_test, data_geth), axis=1)  # luminance 1, color 3, gender 1, ethnicity 1

    traits = ['O', 'C', 'E', 'A', 'S']
    for i, t in enumerate(traits):
        if t == 'A':
            print('uo')
        print('for trait: %s' % t)
        label_trait = labels[:, i]
        reg = LinearRegression().fit(data, label_trait)

        print('reg score: ', reg.score(data, label_trait))

        prediction = reg.predict(data_test)
        loss = np.abs(prediction - labels_test[:, i])
        print('loss: ', np.mean(loss))

        num = P.TEST_LOG.split('_')[-1].split('.')[0]
        save_path = os.path.join(P.LOG_BASE, 'testall_%s_%s.txt' % (num, t))
        with open(save_path, 'a') as mf:
            for i in range(len(loss)):
                line = '%s\n' % str(loss[i])[0:6]
                mf.write(line)

    # for trait: O
    # reg score:  0.03884013482567261
    # loss:  0.115803376
    # for trait: C
    # reg score:  0.031371301668408624
    # loss:  0.12661433
    # for trait: E
    # reg score:  0.05366489112270434
    # loss:  0.12337612
    # uo
    # for trait: A
    # reg score:  0.011632718380941487
    # loss:  0.10779269
    # for trait: S
    # reg score:  0.013905780240829002
    # loss:  0.1263367


# ml_lum_clr_geth_single()