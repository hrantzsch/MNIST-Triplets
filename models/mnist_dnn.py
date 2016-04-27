import chainer
import chainer.functions as F
import chainer.links as L


class MnistDnn(chainer.Chain):

    """Small Dnn smaller."""

    def __init__(self):
        super(MnistDnn, self).__init__(
            conv1=L.Convolution2D(1, 32, 4, stride=2, pad=0),
            bn1=L.BatchNormalization(32),
            conv2=L.Convolution2D(32, 64, 3, stride=1),
            bn2=L.BatchNormalization(64),
            conv3=L.Convolution2D(64, 32, 2, stride=1, pad=0),
            bn3=L.BatchNormalization(32),

            fc1=L.Linear(128, 128),
            fc2=L.Linear(128, 3)
        )
        self.train = True
        self.relu = False

    def clear(self):
        self.loss = None
        self.accuracy = None

    def maybe_relu(self, x):
        if self.relu:
            return F.relu(x)
        else:
            return x

    def __call__(self, x):
        self.clear()

        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(self.maybe_relu(h), 3, stride=2)

        h = self.bn2(self.conv2(h), test=not self.train)
        h = self.maybe_relu(h)

        h = self.bn3(self.conv3(h), test=not self.train)
        h = F.max_pooling_2d(self.maybe_relu(h), 3, stride=2, pad=1)

        h = self.fc1(h)
        h = self.fc2(h)

        return h
