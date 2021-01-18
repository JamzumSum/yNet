from functools import wraps
import torch
class KeyboardInterruptWrapper:
    def __init__(self, solution):
        self._s = solution

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try: return func(*args, **kwargs)
            except KeyboardInterrupt:
                self._s(*args, **kwargs)
        return wrapped

def NoGrad(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapped

def freeze(tensor, f=0.):
    return (1 - f) * tensor + f * tensor.detach()
    
def cal_parameters(model):
    blank = ' '
    print('-'*90)
    print('|'+' '*11+'weight name'+' '*10+'|' \
            +' '*15+'weight shape'+' '*15+'|' \
            +' '*3+'number'+' '*3+'|')
    print('-'*90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4
    
    for key, w_variable in model.named_parameters():
        if len(key) <= 30: 
            key = key + (30-len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40-len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10-len(str_num)) * blank
    
        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-'*90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-'*90)