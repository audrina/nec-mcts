import torch

def config_device():
    # determine if gpu is to be used
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def update_rule(fast_lr):
    def update_q(old, new):
        return old + fast_lr * (new - old)
    return update_q