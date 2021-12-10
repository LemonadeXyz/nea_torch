import torch.optim as opt

def get_optimizer(algo_name, model):

    if algo_name == 'rmsprop':
        optimizer = opt.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-06)
    elif algo_name == 'sgd':
        optimizer = opt.SGD(model.parameters(), lr=0.01, momentum=0, weight_decay=0, nesterov=False)
    elif algo_name == 'adagrad':  # DONE
        optimizer = opt.Adagrad(model.parameters(), lr=0.01, eps=1e-06)
    elif algo_name == 'adadelta':
        optimizer = opt.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-06)
    elif algo_name == 'adam':  # DONE
        optimizer = opt.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    elif algo_name == 'adamax':
        optimizer = opt.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08)
    
    return optimizer


