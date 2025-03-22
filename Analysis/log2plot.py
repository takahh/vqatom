import matplotlib.pyplot as plt

# path = "/Users/taka/PycharmProjects/VQGraph/Analysis/log_2024_11_11"
path = "/Users/taka/Downloads/0322/log"
epoch_num = 16

def plot_loss():
    test_loss_list = []
    with open(path) as f:
        for lines in f.readlines():
            if "test_loss" in lines:
                ele = lines.split()
                test_loss_list.append(float(ele[7]))
    plt.figure()
    plt.plot(test_loss_list)
    plt.show()



def main():
    plot_loss()



if __name__ == '__main__':
    main()