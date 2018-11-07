import numpy as np
import tensorflow as tf
from PIL import Image
import os
import pickle


device = 1
seed = 6
number_latents = 10
path = '/scratch/users/umuguc'
example_latents = np.random.RandomState(seed).randn(number_latents, 512)


def load_network(dev):
    p = '/home/gabras/visualizing-traits/data/karras2018iclr-celebahq-1024x1024.pkl'
    tf.InteractiveSession()
    with tf.device('/gpu:%d' % dev):
        G, D, Gs = pickle.load(open(p, 'rb'))
    return Gs


def face_from_latent(save_image=True):
    with tf.device('/gpu:%d' % device):
        model = load_network(dev=device)

    dummy_label = np.zeros([1] + model.input_shapes[1][1:])

    for i in range(example_latents.shape[0]):
        latent = np.expand_dims(example_latents[i], 0)
        face = model.run(latent, dummy_label)
        face = np.clip(np.rint((face + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
        face = face.transpose((0, 2, 3, 1))  # NCHW => NHWC

        if save_image:
            save_path = os.path.join(path, '%d.png' % i)
            Image.fromarray(face[0], 'RGB').save(save_path)
        else:
            Image.fromarray(face[0], 'RGB').show()


if __name__ == '__main__':
    face_from_latent()
    print('done!')
