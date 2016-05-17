"""A script to train a triplet distance based model of MNIST data.

Very much tailored to mnist...
"""

import numpy as np

import chainer
from chainer import cuda
from chainer import optimizers

from tripletembedding.predictors import TripletNet
from tripletembedding.aux import Logger, load_snapshot

from aux import helpers
from aux.mnist_loader import MnistLoader
from models.mnist_dnn import MnistDnn


args = helpers.get_args()

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
xp = cuda.cupy if args.gpu >= 0 else np

dl = MnistLoader(xp, args.data)
model = TripletNet(MnistDnn)
# if args.initmodel:
#     print("loading pretrained mnist dnn")
#     pretrained = L.Classifier(MnistWithLinear())
#     serializers.load_hdf5(args.initmodel, pretrained)
#     model.dnn.copyparams(pretrained.predictor.dnn)

if args.gpu >= 0:
    model = model.to_gpu()

optimizer = optimizers.MomentumSGD(lr=0.01)
optimizer.setup(model)

if args.initmodel and args.resume:
    load_snapshot(args.initmodel, args.resume, model, optimizer)
    print("Continuing from snapshot. LR: {}".format(optimizer.lr))

logger = Logger(args, optimizer, args.out)

train = list(range(10))
test = list(range(10))
graph_generated = False

for _ in range(1, args.epoch + 1):
    optimizer.new_epoch()
    print('epoch', optimizer.epoch)

    np.random.shuffle(train)
    np.random.shuffle(test)

    margin = 1 + optimizer.epoch ** 2 * 0.001
    print('margin: {:.3}'.format(float(margin)))

    # training
    for anchor in train:
        x_data = dl.get_batch(args.batchsize)
        x = chainer.Variable(x_data)
        optimizer.update(model, x, margin)
        logger.log_iteration("train",
                             float(model.loss.data), float(model.accuracy),
                             float(model.mean_diff), float(model.max_diff))

        if not graph_generated:
            helpers.write_graph(model.loss)
            graph_generated = True

    logger.log_mean("train")

    # testing
    # TODO: need to always test if I want to plot this
    if optimizer.epoch % 5 == 0:
        for anchor in test:
            for _ in range(3):
                x_data = dl.get_batch(args.batchsize, train=False)
                x = chainer.Variable(x_data)
                loss = model(x, margin)
                logger.log_iteration("test",
                                     float(model.loss.data),
                                     float(model.accuracy),
                                     float(model.mean_diff),
                                     float(model.max_diff))
        logger.log_mean("test")

    if optimizer.epoch % args.lrinterval == 0 and optimizer.lr > 0.0000005:
        optimizer.lr *= 0.5
        print("learning rate decreased to {}".format(optimizer.lr))
    if optimizer.epoch % args.interval == 0:
        logger.make_snapshot(model)
