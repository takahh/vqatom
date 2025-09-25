N = 0

path = "/Users/takayukikimura/Downloads/"
adj = f"both_mono/concatenated_adj_batch_{N}.npy"
attr = f"both_mono/concatenated_attr_batch_{N}.npy"
smiles = f"both_mono/smiles_{N}.txt"

import numpy as np


def main():
    # adj_arr = np.load(path + adj).reshape(-1, 100, 100)
    attr_arr = np.load(path + attr).reshape(-1, 100, 27)
    attr_arr = attr_arr[0].reshape(-1)
    attr_arr = attr_arr[attr_arr != 0]

    # print(attr_arr)
    # count = 0
    # with open(path + smiles) as f:
    #     for line in f:
    #         print(line)
    #         count += 1
    #         if count == 1:
    #             break
    c_mask = attr_arr == 6
    print(c_mask)



if __name__ == "__main__":
    main()
