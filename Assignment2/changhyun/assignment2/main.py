import time

from cs231n.classifiers.cnn import VGGNet
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver


# load data
print "loading data..."
data = get_CIFAR10_data()
print "data is now loaded!"

# set up
model = VGGNet(
    weight_scale=0.001,
    reg=0.001
)
print "model is created..."

solver = Solver(model, data,
                num_epochs=1,
                batch_size=50,
                update_rule='adam',
                optim_config={
                    'learning_rate': 1e-3
                },
                verbose=True,
                print_every=20)
print "solver is created..."

# start
start = time.time()
solver._step()
end = time.time()
print "%f second elapsed." % (end - start)
