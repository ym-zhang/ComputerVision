from uwimg import *

def softmax_model(inputs, outputs):
    l = [make_layer(inputs, outputs, SOFTMAX)]
    return make_model(l)

def neural_net(inputs, outputs):
    print (inputs)
    l = [   make_layer(inputs, 64, RELU),
	    make_layer(64, 32, RELU),
            make_layer(32, outputs, SOFTMAX)]
    return make_model(l)

print("loading data...")
train = load_classification_data("cifar.train", "labels.txt", 1)
test  = load_classification_data("cifar.test", "labels.txt", 1)
print("done")
print

print("training model...")
batch = 128
iters = 2000
rate = 0.01
momentum = .9
decay = 0.01

m = neural_net(train.X.cols, train.y.cols)
train_model(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_model(m, train))
print("test accuracy:     %f", accuracy_model(m, test))


