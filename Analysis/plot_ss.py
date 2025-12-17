#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt

CSV_PATH = "ss_matrix.csv"


def load_epoch_mean(path):
    epochs = []
    means = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            means.append(float(row["平均"]))

    return epochs, means


def main():
    epochs, means = load_epoch_mean(CSV_PATH)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, means, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Silhouette Score (non-zero)")
    plt.title("Epoch-wise Mean Silhouette Score")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
