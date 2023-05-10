import torch


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def float_tensor(arr, device='cpu'):
    if device == 'cuda':
        return torch.cuda.FloatTensor(arr)
    else:
        return torch.FloatTensor(arr)
