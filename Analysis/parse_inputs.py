path = "/Users/taka/Downloads/molecules_both_shuffled.npz"
import numpy as np


def main():
    seeinside()


def seeinside():
    arr = np.load(f"{path}")
    for files in arr.files:
        print(files)
        print(arr[files].shape)
        print(arr[files])


if __name__ == '__main__':
    main()