import matplotlib.pyplot as plt


exp_list = ['20000_8', '20000_16', '20000_32', '25000_4', '25000_8', '25000_16', '25000_32', '30000_4',
            '30000_8', '30000_16', '30000_32', '35000_4', '35000_8', '35000_16',
            '35000_32', '40000_8', '40000_16', '40000_32']
# exp_list = ['30000_32']


def get_cbmax_from_log(pair_name):
    effective_cb_size_list = []
    cb_num_mean_list = []
    effective_cb_rate_list = []
    filepath = f'/Users/taka/Documents/vqatom_train_output/{pair_name}/outputs/log'

    if pair_name == '30000_32':
        filepath = f'/Users/taka/Documents/vqatom_train_output/log_bothloss_{pair_name}'
    else:
        import os
        print(pair_name)
        filepath = f'/Users/taka/Documents/vqatom_train_output/bothloss_{pair_name}/outputs/log'
        filepath2 = f'/Users/taka/Documents/vqatom_train_output/bothloss_{pair_name}/0227version/outputs/log'
        try:
            if os.path.getsize(filepath) < 1024:
                filepath = filepath2
        except FileNotFoundError:
            filepath = filepath2
    with open(filepath, 'r') as file:
        for line in file:
            if "observed" in line:
                value = float(line.split(" ")[-1].strip())
                # 全体のCBサイズで割って使用率を出す
                fraction = value / float(pair_name.split('_')[0])
                effective_cb_size_list.append(value)
                effective_cb_rate_list.append(fraction)
            elif "unique_cb_vecs mean" in line:
                value = float(line.split()[10].split(',')[0])
                cb_num_mean_list.append(value)
    max_eff_cb = max(effective_cb_size_list)
    # ebb cb max 時のエポックを取り出す
    idx_eff_cb = effective_cb_size_list.index(max_eff_cb)
    # ebb cb max 時のエポックでの cb_num_mean の値をget
    max_cb_mean = cb_num_mean_list[idx_eff_cb]
    max_cb_rate = effective_cb_rate_list[idx_eff_cb]
    return max_eff_cb, max_cb_mean, max_cb_rate



def plot(data, name):
    plt.figure()
    plt.plot(exp_list, data)
    plt.title(name)
    plt.xticks(rotation=90)  # Rotate x-axis labels by 45 degrees
    plt.tight_layout()
    plt.show()


def main():
    eff_cb_data = []   # size
    eff_cb_rate = []   # rate
    uni_cb_means = []
    for exp in exp_list:
        effective_cb_max, uni_cb_mean_max, effective_cb_rate_max = get_cbmax_from_log(exp)
        eff_cb_data.append(effective_cb_max)
        uni_cb_means.append(uni_cb_mean_max)
        eff_cb_rate.append(effective_cb_rate_max)
    plot(eff_cb_data, "Effective CB Max")
    plot(eff_cb_rate, "Effective CB Rates")
    plot(uni_cb_means, "Unique CB Count Avg")


if __name__ == '__main__':
    main()