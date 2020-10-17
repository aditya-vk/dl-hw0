from uwnet import *

def softmax_model():
    l = [make_connected_layer(784, 10, SOFTMAX)]
    return make_net(l)

def neural_net(h=32):
    l = [
        make_connected_layer(784, h, LRELU),
        make_connected_layer(h, h, LRELU),
        make_connected_layer(h, 10, SOFTMAX)
    ]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels")
test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels")
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

# wider and deeper
# -> 128 -> leaky ReLU -> 128 -> leaky ReLU ->
# training accuracy: 0.9838500022888184
# test accuracy:     0.974399983882904
