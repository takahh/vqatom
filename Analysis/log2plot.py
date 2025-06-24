import matplotlib.pyplot as plt
import numpy as np

# from archive.train_and_eval import train
PATH = "/Users/taka/Documents/log_hptune/bothloss/0227version/outputs/log"
OPATH = "/Users/taka/Documents/plot_hptune/"

# Set global font size
FSIZE = 12
plt.rcParams.update({
    'font.size': FSIZE,           # base font size
    'axes.labelsize': FSIZE,      # x/y labels
    'axes.titlesize': FSIZE,      # title
    'xtick.labelsize': FSIZE,     # x-axis tick labels
    'ytick.labelsize': FSIZE,     # y-axis tick labels
    'legend.fontsize': FSIZE,     # legend text
})


def plot_cb_best(data):
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Organize data by batch size
    grouped = defaultdict(list)
    print(data)
    for key, value in data.items():
        sample_size, batch_size = key.split('_')
        sample_size = int(sample_size)
        batch_size = int(batch_size)
        grouped[batch_size].append((sample_size, value))
        print(key, value)

    # Plot
    plt.figure(figsize=(10, 6))
    for batch_size, values in sorted(grouped.items()):
        values.sort()  # sort by sample_size
        x = [v[0] for v in values]
        y = [v[1] for v in values]
        plt.plot(x, y, marker='o', label=f'Batch size {batch_size}')

    plt.xlabel('Sample size', fontsize=14)
    plt.ylabel('Metric value', fontsize=14)
    plt.title('Metric vs Sample Size for Different Batch Sizes', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_cbmax_from_log(pair_name):
    effective_cb_size_list = []
    cb_num_mean_list = []
    filepath = PATH.replace("bothloss", f"bothloss_{pair_name}")
    with open(filepath, 'r') as file:
        for line in file:
            if "observed" in line:
                effective_cb_size_list.append(float(line.split(" ")[-1].strip()))
            elif "unique_cb_vecs mean" in line:
                cb_num_mean_list.append(float(line.split()[10].split(',')[0]))
    best_epoch = effective_cb_size_list.index(max(effective_cb_size_list))
    return max(effective_cb_size_list), max(cb_num_mean_list), best_epoch


best_unique_cb_num_list = []
best_unique_cb_num_dict = {}
effective_cb_size_list = []
effective_cb_size_dict = {}

def plot(type, num_pair):
    # Data extraction
    epochs = []
    feat_div_loss_train = []
    feat_div_loss_test = []
    cb_loss_train = []
    cb_loss_test = []
    repel_loss_train = []
    repel_loss_test = []
    cb_repel_loss_train = []
    cb_repel_loss_test = []
    sil_loss_train = []
    sil_loss_test = []
    train_loss = []
    unique_cb_mean_list = []
    test_loss = []
    effective_cb_size_list = []
    plt.rcParams.update({'font.size': 24})  # set general font size to 14

    with open(PATH) as file:
    # with open(f'/Users/taka/Documents/vqatom_train_output/{num_pair}', 'r') as file:
        lines = file.readlines()
        print(lines)
        lines = [x for x in lines if "-" in x.split(" ")[1]]

        # lines = [x for x in lines if "repel" not in x and 'unique' not in x]
        for i in range(0, len(lines), 7):

            # epoch_match = lines[i].split('epoch ')[1].split(':')[0]
            # epochs.append(int(epoch_match))

            # Extract losses from both lines
            try:
                line0_parts = lines[i + 1].split(',')
                line1_parts = lines[i + 4].split(' ')
                line2_parts = lines[i + 5].split(' ') # train
                line3_parts = lines[i + 6].split(' ') # test
                # 0       1        2   3  4            5       6         7   8 9             10        11   12 13       14
                # May27 16-43-55: test - feat_div_nega loss:  0.001988, test - commit_loss:  0.000700, test - cb_loss:  0.000700,test - sil_loss:  1.068109,test - repel_loss:  0.570306,
            except IndexError:
                continue
            if "unique_cb_fraction:" in line1_parts:
                continue
            if "unique_cb_fraction:" in line2_parts:
                continue
            if "unique_cb_fraction:" in line3_parts:
                continue

            # Extract specific losses
            train_loss.append(float(line1_parts[5].replace(',', '')))
            try:
                test_loss.append(float(line1_parts[7].strip().replace(",", "")))
            except ValueError:
                test_loss.append(float(line1_parts[7].strip().replace(",", "")
                                       .replace('unique_cb_vecs', '')))
            except IndexError:
                pass
            #
            # May09 03-40-04: epoch 5: loss 0.008738568, test_loss 0.008522105, unique_cb_vecs mean:  596.065625,unique_cb_vecs min:  177.000000,unique_cb_vecs max:  999.000000,
            try:
                unique_cb_mean_list.append(float(line1_parts[11].split(',')[0]))
            except ValueError:
                try:
                    unique_cb_mean_list.append(float(line1_parts[10].split(',')[0]))
                except ValueError:
                    pass
            except IndexError:
                pass
            # Feature divergence loss
            try:
                feat_div_loss_train.append(float(line2_parts[7].replace(',', '')))
            except ValueError:
                try:
                    feat_div_loss_train.append(float(line2_parts[7].replace(',', '')))
                except ValueError:
                    feat_div_loss_train.append(0)
            except IndexError:
                pass
            try:
                feat_div_loss_test.append(float(line3_parts[7].replace(',', '')))
            except ValueError:
                pass
            except IndexError:
                pass

            # CB loss
            try:
                #   0         1           2       3     4                5       6    7            8        9    10             11  12           13       14    15         16   17               18     19         20   21               22    23            24  25
                # ['May27', '10-21-11:', 'train', '-', 'feat_div_nega', 'loss:', '', '0.003996,', 'train', '-', 'commit_loss:', '', '0.000488,', 'train', '-', 'cb_loss:', '', '0.000488,train', '-', 'sil_loss:', '', '1.008963,train', '-', 'repel_loss:', '', '0.644117,\n']
                cb_loss_train.append(float(line2_parts[12].split(',')[0]))
                cb_loss_test.append(float(line3_parts[12].split(',')[0]))
            except IndexError:
                continue
            # Silhouette losses (from two consecutive lines)
            try:
                repel_loss_train.append(float(line2_parts[-5].split(',')[0]))
                repel_loss_test.append(float(line3_parts[-5].split(",")[0]))
            except ValueError:
                pass
            try:
                cb_repel_loss_train.append(float(line2_parts[-1].replace(',', '')))
                cb_repel_loss_test.append(float(line3_parts[-1].replace(',', '')))
            except ValueError:
                pass
            # Silhouette losses (from two consecutive lines)
            try:
                sil_loss_train.append(float(line2_parts[21].split(',')[0]))
                sil_loss_test.append(float(line3_parts[21].split(',')[0]))
            except ValueError:
                pass
            effective_cb_size_list.append(float(line0_parts[0].split(" ")[6].strip()))

    # Create the plot
    plt.figure(figsize=(12, 8))

    epochs = list(range(len(test_loss)))
    # Plot different loss metrics
    if type == 0:
        plt.plot(epochs, train_loss, label='Train Loss', marker='o')
        plt.plot(epochs, test_loss, label='Test Loss', marker='s')
        plt.title('Loss Across Epochs', fontsize=FSIZE)
    elif type == 1:
        # print(feat_div_loss_test)
        plt.plot(epochs, feat_div_loss_train, label='Feature Div Loss Train', marker='^')
        if len(feat_div_loss_test) < 5:
            pass
        else:
            plt.plot(epochs, feat_div_loss_test, label='Feature Div Loss Test', marker='_')
        plt.title('feat_div_loss Across Epochs', fontsize=FSIZE)
    elif type == 2:
        epochs = list(range(len(cb_loss_train)))
        plt.plot(epochs, cb_loss_train, label='CB Loss Train', marker='v')
        epochs = list(range(len(cb_loss_test)))
        plt.plot(epochs, cb_loss_test, label='CB Loss Test', marker='x')
        plt.title('CB loss Across Epochs', fontsize=FSIZE)
    elif type == 3:
        epochs = list(range(len(repel_loss_train)))
        plt.plot(epochs, repel_loss_train, label='Repel Loss Train', marker='d')
        epochs = list(range(len(repel_loss_test)))
        plt.plot(epochs, repel_loss_test, label='Repel Loss Test', marker='p')
        plt.title('Repel loss Across Epochs', fontsize=FSIZE)
    elif type == 4:
        epochs = list(range(len(unique_cb_mean_list)))
        plt.plot(epochs, unique_cb_mean_list, label='Unique CB mean', marker='d')
        plt.title('Unique CB vector counts', fontsize=FSIZE)
        best_unique_cb_num_dict[num_pair] = max(unique_cb_mean_list)
    elif type == 5:
        epochs = list(range(len(cb_repel_loss_train)))
        plt.plot(epochs, cb_repel_loss_train, label='CB Repel Loss Train', marker='d')
        epochs = list(range(len(cb_repel_loss_test)))
        plt.plot(epochs, cb_repel_loss_test, label='CB Repel Loss Test', marker='p')
        plt.title('CB Repel loss Across Epochs', fontsize=FSIZE)
    else:
        epochs = list(range(len(effective_cb_size_list)))
        plt.plot(epochs, effective_cb_size_list, label='Effective CB size', marker='d')
        plt.title('Effective CB size', fontsize=FSIZE)
    print(effective_cb_size_list)
    plt.xlabel('Epoch', labelpad=10, fontsize=FSIZE)
    plt.ylabel('Loss', labelpad=10, fontsize=FSIZE)
    plt.legend(fontsize=17)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')  # Using log scale due to small values
    plt.rcParams.update({'font.size': FSIZE})  # set general font size to 14
    plt.tight_layout()
    plt.savefig(f"/Users/taka/Documents/{type}")
    # # Save the plot
    # plt.savefig('training_metrics_plot.png')
    plt.close()
    return best_unique_cb_num_dict, effective_cb_size_dict

    # print("Plot has been saved as 'training_metrics_plot.png'")


exp_list = ['35000_8', '35000_16', '35000_32', '40000_8', '40000_16', '40000_32', '45000_8', '45000_16', '45000_32']
cb_dict = {}


for exp in exp_list:
    # for type in range(0, 7):
    cb_dict[exp] = get_cbmax_from_log(exp)
print(cb_dict)
exp_list = []
best_epoch_list = []
used_cb_size_list = []
mean_cb_size_list = []
for ele in cb_dict.items():
    exp_list.append(ele[0])

for ele in cb_dict.items():
    used_cb_size_list.append(float(ele[1][0]))
    mean_cb_size_list.append(float(ele[1][1]))
    best_epoch_list.append(float(ele[1][2]))

plt.figure()
plt.plot(exp_list, used_cb_size_list, marker='o', label='Used CB Size')
plt.xlabel("Experiment")
plt.ylabel("Used Codebook Size")
plt.title("Used Codebook Size per Experiment")
plt.xticks(rotation=45, ha='right')  # Rotate for better readability
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.legend()
plt.savefig(f"/Users/taka/Documents/plot_hptune/used_cb_size.png")

plt.figure()
plt.plot(exp_list, mean_cb_size_list, marker='s', color='orange', label='Mean CB Size')
plt.xlabel("Experiment")
plt.ylabel("Mean Codebook Size")
plt.title("Mean Codebook Size per Experiment")
plt.xticks(rotation=90, ha='right')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(f"/Users/taka/Documents/plot_hptune/mean_cb_size.png")

plt.figure()
plt.plot(exp_list, best_epoch_list, marker='s', color='orange', label='Best Epoch')
plt.xlabel("Experiment")
plt.ylabel("Best Epoch")
plt.title("Best Epochs")
plt.xticks(rotation=90, ha='right')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(f"/Users/taka/Documents/plot_hptune/best_epochs.png")


# plot_cb_best(cb_dict)
