import numpy as np

from cs231n.classifiers.cnn import VGGNet
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver


command = "sanity check"

if command == "sanity check":
    # sanity check: loss
    model = VGGNet()

    N = 50
    X = np.random.randn(N, 3, 32, 32)
    y = np.random.randint(10, size=N)

    loss, grads = model.loss(X, y)
    print 'Initial loss (no regularization): ', loss

    model.reg = 0.5
    loss, grads = model.loss(X, y)
    print 'Initial loss (with regularization): ', loss
elif command == "run":
    print "loading data..."
    data = get_CIFAR10_data()

    print "creating model..."
    model = VGGNet(
        fc_dim=1024,
        weight_scale=1e-3,
        reg=0.0
    )

    print "creating solver..."
    solver = Solver(model, data,
                    num_epochs=100,
                    batch_size=256,
                    update_rule='adam',
                    optim_config={
                        'learning_rate': 1e-3
                    })

    print "training has been started..."
    solver.train()
