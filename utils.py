"Utility functions for stuffs"
import numpy as np

def arr_to_seq(arr):
    "Convert nucleotide sequence from numpy array format into string format"
    def _arr_to_seq(row):
        "Convert single nucleotide sequence"
        arr  = row.reshape(-1, 4)
        idxs = arr.argmax(1)
        return ''.join("ACGT"[i] for i in idxs)

    if arr.ndim == 2:
        return np.asarray([_arr_to_seq(row) for row in arr])
    elif arr.ndim == 1:
        return _arr_to_seq(arr)
    else:
        raise Exception("array must have 1 or 2 dimensions")

def seq_to_arr(seq):
    "Convert nucleotide sequence from string format into numpy array"
    def _seq_to_arr(seq):
        "Convert single nucleotide sequence"
        dic = { "A" : np.array([1,0,0,0]),
                "C" : np.array([0,1,0,0]),
                "G" : np.array([0,0,1,0]),
                "T" : np.array([0,0,0,1]) }
        tmp = [dic[c] for c in seq]
        return np.asarray(tmp).flatten()

    if type(seq) is str:
        return _seq_to_arr(seq)
    elif type(seq[0]) in (str, np.string_):
        return np.asarray([_seq_to_arr(s) for s in seq])
    else:
        raise Exception("sequence must either be string or list/np array of strings")

def print_param_shapes(P):
    for p in P.values():
        print str(p) + "\t" + str(p.get_value().shape)

def relu(x):
    "Rectified linear activation function"
    return x * (x > 0)

if __name__=="__main__":
    seq = ["AT", "CG"]
    arr = np.array([[1,0,0,0,0,0,0,1],
                    [0,1,0,0,0,0,1,0]])
    assert (seq_to_arr(seq) == arr).all()
    assert (arr_to_seq(arr) == seq).all()
    assert (seq_to_arr(seq[0]) == arr[0]).all()
    assert arr_to_seq(arr[0]) == seq[0]
