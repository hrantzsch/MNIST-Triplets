import chainer
import chainer.functions as F
import chainer.links as L


class NewCnn(chainer.Chain):

    """Adjusted from AlexBN chainer example."""

    def __init__(self):
        super(NewCnn, self).__init__(
            conv1=L.Convolution2D(1, 256, 11, stride=3, pad=4),
            conv1_reduce=L.Convolution2D(256, 192, 1),
            # max pooling 3x3,2

            conv2=L.Convolution2D(192, 256, 3, stride=1, pad=1),
            conv2_reduce=L.Convolution2D(256, 192, 1),
            # max pooling 3x3,2

            conv3=L.Convolution2D(192, 192, 3, stride=1, pad=1),
            conv3_reduce=L.Convolution2D(192, 96, 1),
            # max pooling 3x3,2

            conv4=L.Convolution2D(96, 32, 3, stride=1, pad=1),
            # avg pooling 5x5,1

            fc1=L.Linear(32, 32)
        )

    def __call__(self, x):

        h = self.conv1(x)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.conv1_reduce(h)

        h = self.conv2(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.conv2_reduce(h)

        h = self.conv3(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.conv3_reduce(h)

        h = self.conv4(h)
        h = F.average_pooling_2d(h, 3, stride=1)

        h = self.fc1(h)

        return h


# class NewCnn(chainer.Chain):
#
#     """Adjusted from AlexBN chainer example."""
#
#     def __init__(self):
#         super(NewCnn, self).__init__(
#             conv1=L.Convolution2D(1, 256, 7, stride=2, pad=3),
#             conv1_reduce=L.Convolution2D(256, 192, 1),
#             # max pooling 3x3,2
#
#             conv2=L.Convolution2D(192, 256, 3, stride=1, pad=1),
#             # max pooling 3x3,2
#             conv2_reduce=L.Convolution2D(256, 192, 1),
#
#             conv3=L.Convolution2D(192, 192, 3, stride=1, pad=1),
#             # max pooling 3x3,2
#             conv3_reduce=L.Convolution2D(192, 96, 1),
#
#             conv4=L.Convolution2D(96, 32, 3, stride=1, pad=1),
#             # avg pooling 5x5,1
#
#             fc1=L.Linear(32, 32)
#         )
#
#     def __call__(self, x):
#
#         h = self.conv1(x)
#         h = F.max_pooling_2d(h, 3, stride=2)
#         h = self.conv1_reduce(h)
#
#         h = self.conv2(h)
#         h = F.max_pooling_2d(h, 3, stride=2)
#         h = self.conv2_reduce(h)
#
#         h = self.conv3(h)
#         h = F.max_pooling_2d(h, 3, stride=2)
#         h = self.conv3_reduce(h)
#
#         h = self.conv4(h)
#         h = F.average_pooling_2d(h, 5, stride=1)
#
#         h = self.fc1(h)
#
#         return h
