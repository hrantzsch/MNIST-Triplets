import chainer
import chainer.functions as F
import chainer.links as L


class MnistDnn(chainer.Chain):

    """Small Dnn smaller."""

    def __init__(self):
        super(MnistDnn, self).__init__(
            conv1=L.Convolution2D(1, 32, 4, stride=2, pad=0),
            conv2=L.Convolution2D(32, 64, 3, stride=1),
            conv3=L.Convolution2D(64, 32, 2, stride=1, pad=0),

            fc1=L.Linear(128, 128),
            fc2=L.Linear(128, 3)
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.conv2(h)

        h = self.conv3(h)
        h = F.max_pooling_2d(h, 3, stride=2, pad=1)

        h = self.fc1(h)
        h = self.fc2(h)

        return h
