# Salvando uma instância de classe em arquivo
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# Restaurando objeto a partir de arquivo '.pkl'
def read_object(obj_path):
    with open(obj_path, 'rb') as pkl_obj:
        obj = pickle.load(pkl_obj)
        return obj