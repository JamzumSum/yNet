
def cal_parameters(model):
    blank = ' '
    print('-'*90)
    print('|'+' '*11+'weight name'+' '*10+'|'
          + ' '*15+'weight shape'+' '*15+'|'
            + ' '*3+'number'+' '*3+'|')
    print('-'*90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

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
    print('The parameters of Model {}: {:4f}M'.format(
        model._get_name(), num_para * type_size / 1000 / 1000))
    print('-'*90)

def update_default(default, update, copy=False):
    if copy: default = default.copy()
    default.update(update)
    return default

def soft_update(default, update):
    f = {dict: dict.items, list: enumerate}[type(update)]
    for k, v in f(update):
        if k in default:
            if isinstance(v, dict) and isinstance(default[k], dict): 
                default[k] = soft_update(default[k], v)
            elif isinstance(v, list) and isinstance(default[k], list): 
                default[k] = soft_update(default[k], v)
            else: default[k] = v
        else: default[k] = v
    return default