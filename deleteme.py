import re
import matplotlib.pyplot as plt
import torch

log = """Feb18 02-53-47: [ENT] epoch=1 key=6_0_3_1_1_0 H=5.8720 effK=1113 max_p=0.0303 ent_w=0.001 contrib=-0.005872
Feb18 02-55-15: [ENT] epoch=1 key=6_0_3_1_1_0 H=5.9172 effK=625 max_p=0.0153 ent_w=0.001 contrib=-0.005917
Feb18 02-56-44: [ENT] epoch=1 key=6_0_3_1_1_0 H=5.9272 effK=597 max_p=0.0121 ent_w=0.001 contrib=-0.005927
Feb18 02-58-10: [ENT] epoch=1 key=6_0_3_1_1_0 H=5.9833 effK=620 max_p=0.0104 ent_w=0.001 contrib=-0.005983
Feb18 02-59-38: [ENT] epoch=1 key=6_0_3_1_1_0 H=6.0068 effK=625 max_p=0.0107 ent_w=0.001 contrib=-0.006007
Feb18 03-01-04: [ENT] epoch=1 key=6_0_3_1_1_0 H=5.9735 effK=622 max_p=0.0112 ent_w=0.001 contrib=-0.005974
Feb18 03-02-30: [ENT] epoch=1 key=6_0_3_1_1_0 H=5.9900 effK=615 max_p=0.0135 ent_w=0.001 contrib=-0.005990
Feb18 03-03-56: [ENT] epoch=1 key=6_0_3_1_1_0 H=5.9888 effK=619 max_p=0.0128 ent_w=0.001 contrib=-0.005989
Feb18 03-05-21: [ENT] epoch=1 key=6_0_3_1_1_0 H=5.9770 effK=623 max_p=0.0121 ent_w=0.001 contrib=-0.005977
Feb18 03-06-49: [ENT] epoch=1 key=6_0_3_1_1_0 H=5.9882 effK=615 max_p=0.0115 ent_w=0.001 contrib=-0.005988
Feb18 03-08-15: [ENT] epoch=1 key=6_0_3_1_1_0 H=5.9964 effK=622 max_p=0.0102 ent_w=0.001 contrib=-0.005996
Feb18 03-09-41: [ENT] epoch=1 key=6_0_3_1_1_0 H=5.9719 effK=618 max_p=0.0124 ent_w=0.001 contrib=-0.005972
Feb18 03-19-25: [ENT] epoch=2 key=6_0_3_1_1_0 H=5.9552 effK=612 max_p=0.0161 ent_w=0.001 contrib=-0.005955
Feb18 03-20-51: [ENT] epoch=2 key=6_0_3_1_1_0 H=5.9881 effK=617 max_p=0.0131 ent_w=0.001 contrib=-0.005988
Feb18 03-22-18: [ENT] epoch=2 key=6_0_3_1_1_0 H=5.9808 effK=603 max_p=0.0111 ent_w=0.001 contrib=-0.005981
Feb18 03-23-46: [ENT] epoch=2 key=6_0_3_1_1_0 H=5.9921 effK=612 max_p=0.0109 ent_w=0.001 contrib=-0.005992
Feb18 03-25-12: [ENT] epoch=2 key=6_0_3_1_1_0 H=5.9139 effK=595 max_p=0.0225 ent_w=0.001 contrib=-0.005914
Feb18 03-26-40: [ENT] epoch=2 key=6_0_3_1_1_0 H=5.9673 effK=615 max_p=0.0133 ent_w=0.001 contrib=-0.005967
Feb18 03-28-07: [ENT] epoch=2 key=6_0_3_1_1_0 H=6.0079 effK=608 max_p=0.0122 ent_w=0.001 contrib=-0.006008
Feb18 03-29-34: [ENT] epoch=2 key=6_0_3_1_1_0 H=6.0001 effK=612 max_p=0.0121 ent_w=0.001 contrib=-0.006000
Feb18 03-30-59: [ENT] epoch=2 key=6_0_3_1_1_0 H=6.0205 effK=617 max_p=0.0100 ent_w=0.001 contrib=-0.006020
Feb18 03-32-25: [ENT] epoch=2 key=6_0_3_1_1_0 H=5.9394 effK=603 max_p=0.0219 ent_w=0.001 contrib=-0.005939
Feb18 03-33-51: [ENT] epoch=2 key=6_0_3_1_1_0 H=5.9154 effK=598 max_p=0.0179 ent_w=0.001 contrib=-0.005915
Feb18 03-35-18: [ENT] epoch=2 key=6_0_3_1_1_0 H=5.9552 effK=607 max_p=0.0195 ent_w=0.001 contrib=-0.005955
Feb18 03-44-54: [ENT] epoch=3 key=6_0_3_1_1_0 H=5.9449 effK=612 max_p=0.0137 ent_w=0.001 contrib=-0.005945
Feb18 03-46-21: [ENT] epoch=3 key=6_0_3_1_1_0 H=5.9290 effK=604 max_p=0.0193 ent_w=0.001 contrib=-0.005929
Feb18 03-47-47: [ENT] epoch=3 key=6_0_3_1_1_0 H=5.9941 effK=600 max_p=0.0107 ent_w=0.001 contrib=-0.005994
Feb18 03-49-14: [ENT] epoch=3 key=6_0_3_1_1_0 H=6.0139 effK=607 max_p=0.0122 ent_w=0.001 contrib=-0.006014
Feb18 03-50-41: [ENT] epoch=3 key=6_0_3_1_1_0 H=6.0173 effK=616 max_p=0.0114 ent_w=0.001 contrib=-0.006017
Feb18 03-52-09: [ENT] epoch=3 key=6_0_3_1_1_0 H=6.0080 effK=620 max_p=0.0110 ent_w=0.001 contrib=-0.006008
Feb18 03-53-34: [ENT] epoch=3 key=6_0_3_1_1_0 H=6.0327 effK=616 max_p=0.0097 ent_w=0.001 contrib=-0.006033
Feb18 03-54-58: [ENT] epoch=3 key=6_0_3_1_1_0 H=5.9895 effK=611 max_p=0.0175 ent_w=0.001 contrib=-0.005989
Feb18 03-56-24: [ENT] epoch=3 key=6_0_3_1_1_0 H=5.9695 effK=612 max_p=0.0160 ent_w=0.001 contrib=-0.005970
Feb18 03-57-50: [ENT] epoch=3 key=6_0_3_1_1_0 H=6.0041 effK=615 max_p=0.0160 ent_w=0.001 contrib=-0.006004
Feb18 03-59-17: [ENT] epoch=3 key=6_0_3_1_1_0 H=5.9963 effK=620 max_p=0.0137 ent_w=0.001 contrib=-0.005996
Feb18 04-00-44: [ENT] epoch=3 key=6_0_3_1_1_0 H=5.9275 effK=602 max_p=0.0227 ent_w=0.001 contrib=-0.005927
Feb18 04-10-19: [ENT] epoch=4 key=6_0_3_1_1_0 H=5.9264 effK=602 max_p=0.0149 ent_w=0.001 contrib=-0.005926
Feb18 04-11-47: [ENT] epoch=4 key=6_0_3_1_1_0 H=5.8165 effK=588 max_p=0.0234 ent_w=0.001 contrib=-0.005817
Feb18 04-13-13: [ENT] epoch=4 key=6_0_3_1_1_0 H=6.0146 effK=606 max_p=0.0096 ent_w=0.001 contrib=-0.006015
Feb18 04-14-41: [ENT] epoch=4 key=6_0_3_1_1_0 H=5.9256 effK=592 max_p=0.0156 ent_w=0.001 contrib=-0.005926
Feb18 04-16-09: [ENT] epoch=4 key=6_0_3_1_1_0 H=5.9215 effK=600 max_p=0.0188 ent_w=0.001 contrib=-0.005921
Feb18 04-17-35: [ENT] epoch=4 key=6_0_3_1_1_0 H=5.8490 effK=586 max_p=0.0226 ent_w=0.001 contrib=-0.005849
Feb18 04-19-01: [ENT] epoch=4 key=6_0_3_1_1_0 H=5.9106 effK=594 max_p=0.0170 ent_w=0.001 contrib=-0.005911
Feb18 04-20-27: [ENT] epoch=4 key=6_0_3_1_1_0 H=5.9644 effK=605 max_p=0.0134 ent_w=0.001 contrib=-0.005964
Feb18 04-21-53: [ENT] epoch=4 key=6_0_3_1_1_0 H=5.9664 effK=608 max_p=0.0116 ent_w=0.001 contrib=-0.005966
Feb18 04-23-20: [ENT] epoch=4 key=6_0_3_1_1_0 H=5.8594 effK=589 max_p=0.0187 ent_w=0.001 contrib=-0.005859
Feb18 04-24-49: [ENT] epoch=4 key=6_0_3_1_1_0 H=6.0114 effK=616 max_p=0.0108 ent_w=0.001 contrib=-0.006011
Feb18 04-26-14: [ENT] epoch=4 key=6_0_3_1_1_0 H=5.9229 effK=602 max_p=0.0217 ent_w=0.001 contrib=-0.005923
Feb18 04-35-51: [ENT] epoch=5 key=6_0_3_1_1_0 H=5.7883 effK=571 max_p=0.0229 ent_w=0.001 contrib=-0.005788
Feb18 04-37-18: [ENT] epoch=5 key=6_0_3_1_1_0 H=5.8814 effK=591 max_p=0.0207 ent_w=0.001 contrib=-0.005881
Feb18 04-38-44: [ENT] epoch=5 key=6_0_3_1_1_0 H=5.9226 effK=594 max_p=0.0156 ent_w=0.001 contrib=-0.005923
Feb18 04-40-11: [ENT] epoch=5 key=6_0_3_1_1_0 H=5.9581 effK=596 max_p=0.0145 ent_w=0.001 contrib=-0.005958
Feb18 04-41-36: [ENT] epoch=5 key=6_0_3_1_1_0 H=5.8553 effK=582 max_p=0.0194 ent_w=0.001 contrib=-0.005855
Feb18 04-43-03: [ENT] epoch=5 key=6_0_3_1_1_0 H=5.9785 effK=610 max_p=0.0156 ent_w=0.001 contrib=-0.005979
Feb18 04-44-28: [ENT] epoch=5 key=6_0_3_1_1_0 H=5.8631 effK=587 max_p=0.0174 ent_w=0.001 contrib=-0.005863
Feb18 04-45-53: [ENT] epoch=5 key=6_0_3_1_1_0 H=5.9652 effK=603 max_p=0.0112 ent_w=0.001 contrib=-0.005965
Feb18 04-47-19: [ENT] epoch=5 key=6_0_3_1_1_0 H=5.7504 effK=578 max_p=0.0204 ent_w=0.001 contrib=-0.005750
Feb18 04-48-45: [ENT] epoch=5 key=6_0_3_1_1_0 H=5.8927 effK=589 max_p=0.0162 ent_w=0.001 contrib=-0.005893
Feb18 04-50-10: [ENT] epoch=5 key=6_0_3_1_1_0 H=5.9347 effK=602 max_p=0.0155 ent_w=0.001 contrib=-0.005935
Feb18 04-51-35: [ENT] epoch=5 key=6_0_3_1_1_0 H=5.5606 effK=539 max_p=0.0292 ent_w=0.001 contrib=-0.005561
Feb18 05-01-07: [ENT] epoch=6 key=6_0_3_1_1_0 H=5.5836 effK=540 max_p=0.0265 ent_w=0.001 contrib=-0.005584
Feb18 05-02-33: [ENT] epoch=6 key=6_0_3_1_1_0 H=5.8111 effK=582 max_p=0.0223 ent_w=0.001 contrib=-0.005811
Feb18 05-03-59: [ENT] epoch=6 key=6_0_3_1_1_0 H=5.4370 effK=497 max_p=0.0301 ent_w=0.001 contrib=-0.005437
Feb18 05-05-25: [ENT] epoch=6 key=6_0_3_1_1_0 H=5.7583 effK=569 max_p=0.0181 ent_w=0.001 contrib=-0.005758
Feb18 05-06-51: [ENT] epoch=6 key=6_0_3_1_1_0 H=5.8311 effK=582 max_p=0.0200 ent_w=0.001 contrib=-0.005831
Feb18 05-08-17: [ENT] epoch=6 key=6_0_3_1_1_0 H=5.8453 effK=583 max_p=0.0216 ent_w=0.001 contrib=-0.005845
Feb18 05-09-42: [ENT] epoch=6 key=6_0_3_1_1_0 H=5.8713 effK=583 max_p=0.0171 ent_w=0.001 contrib=-0.005871
Feb18 05-11-07: [ENT] epoch=6 key=6_0_3_1_1_0 H=5.9041 effK=591 max_p=0.0186 ent_w=0.001 contrib=-0.005904
Feb18 05-12-30: [ENT] epoch=6 key=6_0_3_1_1_0 H=5.9535 effK=600 max_p=0.0163 ent_w=0.001 contrib=-0.005953
Feb18 05-13-56: [ENT] epoch=6 key=6_0_3_1_1_0 H=5.9555 effK=595 max_p=0.0143 ent_w=0.001 contrib=-0.005956
Feb18 05-15-21: [ENT] epoch=6 key=6_0_3_1_1_0 H=5.7001 effK=568 max_p=0.0206 ent_w=0.001 contrib=-0.005700
Feb18 05-16-48: [ENT] epoch=6 key=6_0_3_1_1_0 H=5.8944 effK=592 max_p=0.0219 ent_w=0.001 contrib=-0.005894
Feb18 05-26-44: [ENT] epoch=7 key=6_0_3_1_1_0 H=5.8826 effK=588 max_p=0.0155 ent_w=0.001 contrib=-0.005883
Feb18 05-28-09: [ENT] epoch=7 key=6_0_3_1_1_0 H=5.7130 effK=560 max_p=0.0169 ent_w=0.001 contrib=-0.005713
Feb18 05-29-38: [ENT] epoch=7 key=6_0_3_1_1_0 H=5.8722 effK=588 max_p=0.0192 ent_w=0.001 contrib=-0.005872
Feb18 05-31-06: [ENT] epoch=7 key=6_0_3_1_1_0 H=5.5606 effK=522 max_p=0.0271 ent_w=0.001 contrib=-0.005561
Feb18 05-32-34: [ENT] epoch=7 key=6_0_3_1_1_0 H=5.9286 effK=591 max_p=0.0145 ent_w=0.001 contrib=-0.005929
Feb18 05-34-02: [ENT] epoch=7 key=6_0_3_1_1_0 H=5.7376 effK=557 max_p=0.0210 ent_w=0.001 contrib=-0.005738
Feb18 05-35-30: [ENT] epoch=7 key=6_0_3_1_1_0 H=5.4875 effK=497 max_p=0.0284 ent_w=0.001 contrib=-0.005488
Feb18 05-36-57: [ENT] epoch=7 key=6_0_3_1_1_0 H=5.6373 effK=538 max_p=0.0190 ent_w=0.001 contrib=-0.005637
Feb18 05-38-23: [ENT] epoch=7 key=6_0_3_1_1_0 H=5.5458 effK=520 max_p=0.0260 ent_w=0.001 contrib=-0.005546
Feb18 05-39-49: [ENT] epoch=7 key=6_0_3_1_1_0 H=5.6476 effK=533 max_p=0.0244 ent_w=0.001 contrib=-0.005648
Feb18 05-41-15: [ENT] epoch=7 key=6_0_3_1_1_0 H=5.8164 effK=563 max_p=0.0159 ent_w=0.001 contrib=-0.005816
Feb18 05-42-44: [ENT] epoch=7 key=6_0_3_1_1_0 H=5.4871 effK=505 max_p=0.0278 ent_w=0.001 contrib=-0.005487
Feb18 05-52-24: [ENT] epoch=8 key=6_0_3_1_1_0 H=5.7914 effK=559 max_p=0.0185 ent_w=0.001 contrib=-0.005791
Feb18 05-53-50: [ENT] epoch=8 key=6_0_3_1_1_0 H=5.7961 effK=561 max_p=0.0190 ent_w=0.001 contrib=-0.005796
Feb18 05-55-18: [ENT] epoch=8 key=6_0_3_1_1_0 H=5.6646 effK=535 max_p=0.0266 ent_w=0.001 contrib=-0.005665
Feb18 05-56-44: [ENT] epoch=8 key=6_0_3_1_1_0 H=5.7989 effK=566 max_p=0.0190 ent_w=0.001 contrib=-0.005799
Feb18 05-58-11: [ENT] epoch=8 key=6_0_3_1_1_0 H=5.7075 effK=532 max_p=0.0201 ent_w=0.001 contrib=-0.005708
Feb18 05-59-35: [ENT] epoch=8 key=6_0_3_1_1_0 H=5.5591 effK=516 max_p=0.0241 ent_w=0.001 contrib=-0.005559
Feb18 06-01-01: [ENT] epoch=8 key=6_0_3_1_1_0 H=5.5913 effK=524 max_p=0.0202 ent_w=0.001 contrib=-0.005591"""

epochs = []
Hs = []
effKs = []
max_ps = []
effKs = []

for line in log.splitlines():
    if "[ENT]" not in line:
        continue
    ep = int(re.search(r"epoch=(\d+)", line).group(1))
    H  = float(re.search(r"\bH=([0-9.]+)", line).group(1))
    effK = int(re.search(r"effK=(\d+)", line).group(1))
    max_p = float(re.search(r"max_p=([0-9.]+)", line).group(1))

    epochs.append(ep)
    Hs.append(H)
    effKs.append(effK)
    max_ps.append(max_p)

plt.figure()
plt.plot(Hs)
plt.title("H (entropy) over steps")
plt.xlabel("step")
plt.ylabel("H")
plt.show()

plt.figure()
plt.plot(max_ps)
plt.title("max_p over steps")
plt.xlabel("step")
plt.ylabel("max_p")
plt.show()

plt.figure()
plt.plot(effKs)
plt.title("effK over steps")
plt.xlabel("step")
plt.ylabel("effK")
plt.show()
