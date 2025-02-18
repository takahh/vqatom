import matplotlib.pyplot as plt

# path = "/Users/taka/PycharmProjects/VQGraph/Analysis/log_2024_11_11"
path = "/Users/taka/PycharmProjects/VQGraph2/Analysis/log"
epoch_num = 16


def get_four_lists(kwd):
    with open(path) as f:
        line_list = [x for x in f.readlines() if kwd in x]
    # train_known_g, epoch   1, feature_loss:  0.457287| edge_loss:  0.396768| commit_loss:  0.230803
    # test_known_g, epoch   1, feature_loss:  0.364672| edge_loss:  0.492721| commit_loss:  0.207531
    # test_unknown_g, epoch   1, feature_loss:  0.217055| edge_loss:  0.628982| commit_loss:  0.112391
    # accuracy, epoch   1, loss_total 0.9584
    if kwd == "accuracy":
        # accuracy, epoch  38, tran_acc: 0.8737903237342834, ind_acc: 0.7496504783630371
        ind_acc_list = [float(x.split()[4][:-1]) for x in line_list]
        return ind_acc_list
    else:
        # Nov10 22-46-18: train_known_g, epoch   2, feature_loss:  0.001953| edge_loss:  0.018125| commit_loss:  0.251250, loss_train 2.4667
        feat_loss_list = [float(x.split()[4][:-1]) for x in line_list]
        edge_loss_list = [float(x.split()[6][:-1]) for x in line_list]
        commit_loss_list = [float(x.split()[8][:-1]) for x in line_list]
        # feat_loss_list = [float(x.split()[6][:-1]) for x in line_list]
        # edge_loss_list = [float(x.split()[8][:-1]) for x in line_list]
        # commit_loss_list = [float(x.split()[10][:-1]) for x in line_list]
        # model_loss_list = [float(x.split()[-1].strip()) for x in line_list]
        return feat_loss_list, edge_loss_list, commit_loss_list


def main():
    with open(path) as f:
        train_feat_loss_list, train_edge_loss_list, train_commit_loss_list = get_four_lists("train_known")
        tran_feat_loss_list, tran_edge_loss_list, tran_commit_loss_list = get_four_lists("test_known")
        ind_feat_loss_list, ind_edge_loss_list, ind_commit_loss_list = get_four_lists("test_unknown")
        ind_acc_list = get_four_lists("accuracy")

        def plot_three(train_list, plotname, tran_list=None, ind_list=None):
            plt.figure()
            plt.title(plotname)
            # plt.ylim(0, 0.20)
            if plotname == "Ind Accuracy":
                plt.scatter(list(range(epoch_num)), ind_acc_list, label='ind')
            else:
                plt.scatter(list(range(epoch_num)), train_list, label='train')
                plt.scatter(list(range(epoch_num)), tran_list, label='tran')
                plt.scatter(list(range(epoch_num)), ind_list, label='ind')
            plt.legend()
            plt.show()

        plot_three(train_feat_loss_list, "Feature Loss", tran_feat_loss_list, ind_feat_loss_list)
        plot_three(train_edge_loss_list, "Edge Loss", tran_edge_loss_list, ind_edge_loss_list)
        plot_three(train_commit_loss_list, "Commit Loss", tran_commit_loss_list, ind_commit_loss_list)
        plot_three(train_model_loss_list, "Model Loss", tran_model_loss_list, ind_model_loss_list)
        plot_three(ind_acc_list, "Ind Accuracy")


if __name__ == '__main__':
    main()