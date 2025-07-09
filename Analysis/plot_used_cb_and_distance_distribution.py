import matplotlib.pyplot as plt
import numpy as np
from sympy.physics.units import force

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})
data = [
    (155, [244251072.0, 608840.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (1406, [19129118.0, 38743140.0, 67904832.0, 94160528.0, 24838412.0, 83872.0, 0.0, 0.0, 0.0, 0.0]),
    (2506, [7801178.0, 20598714.0, 35200052.0, 56072228.0, 70987808.0, 48483960.0, 5715968.0, 0.0, 0.0, 0.0]),
    (3486, [5319830.0, 14294552.0, 24155492.0, 37778360.0, 52991708.0, 57606188.0, 47459976.0, 5253792.0, 0.0, 0.0]),
    (3817, [4290218.0, 11371824.0, 18655802.0, 27785756.0, 40022020.0, 48815944.0, 50196200.0, 40802664.0, 2919474.0, 0.0]),
    (3964, [3676224.0, 9405734.0, 15183176.0, 21953720.0, 30260784.0, 39676020.0, 44330128.0, 45781600.0, 33439956.0, 1152561.0]),
]

plt.figure()
peak_xy_list = []

for i, (shape_size, values) in enumerate(data):
    if i in [0, 2, 5]:
        plt.plot(values, marker='.', label=f"{i}")

for i, (shape_size, values) in enumerate(data):
    max_val = max(values)
    max_x = values.index(max_val)
    peak_xy_list.append((max_x + 0.4, max_val - 0.5e8))

xtick_pos = list(range(10))
xtick_labels = ["0 – 1.5", "1.5 – 3.0", "3.0 – 4.5", "4.5 – 6.0", "6.0 – 7.5",
                "7.5 – 9.0", "9.0 – 10.5", "10.5 – 12.0", "12.0 – 13.5", "13.5 – 15.0"]

plt.xticks(ticks=xtick_pos, labels=xtick_labels, rotation=90)
plt.xlabel("Distance Range", labelpad=10)
plt.ylabel("Frequency", labelpad=10)

# Annotate each peak
for i, ((shape_size, _), (x, y)) in enumerate(zip(data, peak_xy_list)):
    if i == 0:
        plt.annotate(f"codebook size\n  {shape_size}",
                 xy=(x, y),
                 xytext=(0, 10),
                 textcoords='offset points',
                 fontsize=12)
    elif i in [2, 5]:
        plt.annotate(f"{shape_size}",
                 xy=(x-0.2, y+0.3e8),
                 xytext=(0, 10),
                 textcoords='offset points',
                 fontsize=12)
    else:
        pass

plt.legend(title="epoch")
# plt.grid(True)
plt.tight_layout()
plt.show()
