"""Help loading data from mnist pkl files.
Completely tailored to my locally generated files..."""
import os
import pickle
import numpy as np
from scipy.misc import imread


class MnistLoader:

    def __init__(self, xp, data):
        TRAIN = os.path.join(data, "train")
        TEST = os.path.join(data, "test")
        self.train_group = {int(d[:4]): [os.path.join(TRAIN, d, s)
                            for s in os.listdir(os.path.join(TRAIN, d))]
                            for d in os.listdir(TRAIN)}
        self.test_group = {int(d[:4]): [os.path.join(TEST, d, s)
                           for s in os.listdir(os.path.join(TEST, d))]
                           for d in os.listdir(TEST)}
        self.xp = xp

    def get_rnd_triplet(self, group, anchor=None):
        classes = list(range(len(group)))
        if anchor is None:
            np.random.shuffle(classes)
            anchor = classes.pop()
        else:
            classes.remove(anchor)
        negative = np.random.choice(classes)

        return (group[anchor][np.random.randint(len(group[anchor]))],
                group[anchor][np.random.randint(len(group[anchor]))],
                group[negative][np.random.randint(len(group[negative]))])

    def get_batch(self, batchsize, anchor=None, train=True):
        group = self.train_group if train else self.test_group

        triplets = [self.get_rnd_triplet(group, anchor)
                    for _ in range(batchsize)]
        paths = []
        for i in range(3):
            for j in range(batchsize):
                paths.append(triplets[j][i])

        batch = self.xp.array([imread(path).astype(self.xp.float32)
                              for path in paths], dtype=self.xp.float32)
        return (batch / 255.0)[:, self.xp.newaxis, ...]
