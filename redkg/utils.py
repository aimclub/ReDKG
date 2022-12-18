import pickle

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
def pickle_dump(file_path, file):
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)
    # print(f'Logging Info - Saved: {file_path}')


def pickle_load(file_path):
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        # print(f'Logging Info - Loaded: {file_path}')
    except EOFError:
        # print(f'Logging Error - Cannot load: {file_path}')
        obj = None
    return obj
