from uwnet import *

def softmax_model():
    l = [make_connected_layer(3072, 10, SOFTMAX)]
    return make_net(l)

def neural_net(h=32):
    l = [
        make_connected_layer(3072, h, LRELU),
        make_connected_layer(h, h, LRELU),
        make_connected_layer(h, 10, SOFTMAX)
    ]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test", "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .0

m = neural_net(h=128)
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")

print("evaluating model...")
print("training accuracy:", accuracy_net(m, train))
print("test accuracy:    ", accuracy_net(m, test))

# wider and deeper: still better (although not as good as MNIST)
# -> 128 -> leaky ReLU -> 128 -> leaky ReLU ->
# training accuracy: 0.5445600152015686
# test accuracy:     0.503600001335144
