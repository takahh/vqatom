import json
import glob
import os
import re
import argparse
import matplotlib.pyplot as plt


def get_epoch_from_name(path):
    m = re.search(r"epoch_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.json_dir, "epoch_*.json"))
    files = sorted(files, key=get_epoch_from_name)

    if not files:
        print(f"No epoch json found in: {args.json_dir}")
        return

    epochs = []

    train_auc = []
    valid_auc = []
    final_auc = []

    loss_total = []
    loss_cls = []
    loss_reg = []
    loss_contact = []

    for fp in files:
        with open(fp, "r") as f:
            d = json.load(f)

        ep = d.get("epoch", get_epoch_from_name(fp))
        epochs.append(ep)

        train_metrics = d.get("train_metrics", {})
        valid_metrics = d.get("valid_metrics", {})
        final_metrics = d.get("final_eval_metrics", {})

        train_stat = d.get("train_stat", {})

        train_auc.append(train_metrics.get("auroc"))
        valid_auc.append(valid_metrics.get("auroc"))
        final_auc.append(final_metrics.get("auroc"))

        loss_total.append(train_stat.get("loss"))
        loss_cls.append(train_stat.get("loss_cls"))
        loss_reg.append(train_stat.get("loss_reg"))
        loss_contact.append(train_stat.get("loss_contact"))

    # AUC
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_auc, marker="o", label="train auc")
    plt.plot(epochs, valid_auc, marker="o", label="valid auc")
    plt.plot(epochs, final_auc, marker="o", label="final auc")
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title("AUROC vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.json_dir, "auc_curve.png"))
    plt.show()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_total, marker="o", label="loss total")
    plt.plot(epochs, loss_cls, marker="o", label="loss cls")
    plt.plot(epochs, loss_reg, marker="o", label="loss reg")
    plt.plot(epochs, loss_contact, marker="o", label="loss contact")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.json_dir, "loss_curve.png"))
    plt.show()


if __name__ == "__main__":
    main()