import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, auc, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def get_accuracy_per_class(results):
    accuracies = {}
    for label in ["Non-Damage", "Earthquake", "Fire", "Flood"]:
        total = results[results["True"] == label]
        correct = total[total["Correct"] == True]
        acc = len(correct) / len(total)
        accuracies[label] = acc
    
    bar_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    fig, ax = plt.subplots()
    bar_container = ax.bar(accuracies.keys(), accuracies.values(), color=bar_colors)
    ax.set(ylabel="Accuracy", xlabel="Natural Disaster", title="Accuracy per Class")
    ax.bar_label(bar_container, fmt="{:,.3f}")
    plt.show()


def calculate_precision(y_true, y_pred, average=None):
    return precision_score(y_true=y_true, y_pred=y_pred, average=average)

def calculate_recall(y_true, y_pred, average=None):
    return recall_score(y_true=y_true, y_pred=y_pred, average=average)


def plot_roc_curve(label_names, y_true, y_score):
    # See https://vitalflux.com/roc-curve-auc-python-false-positive-true-positive-rate/ for reference
    y_true_bin = label_binarize(y_true, classes=label_names)
    plt.figure(figsize=(10, 6))

    for i, name in enumerate(label_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label="Random Classifier")   
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='green', label="Perfect Classifier")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc="lower right")
    plt.show()


def get_confusion_matrix(y_true, y_pred, label_names):

    # Refer to https://stackoverflow.com/questions/65618137/confusion-matrix-for-multiple-classes-in-python for reference
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt = 'g')

    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=0)
    ax.xaxis.set_ticklabels(label_names, fontsize=10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(label_names, fontsize=10)
    plt.yticks(rotation=0)

    plt.title("Confusion Matrix", fontsize=20)
    plt.show()

def main():
    label_names = ["Non-Damage", "Earthquake", "Fire", "Flood"]

    df = pd.read_csv("resnet50_results.csv")
    df["Correct"] = df["Predicted"] == df["True"]

    precision = calculate_precision(df["True"], df["Predicted"], average="weighted")
    recall = calculate_recall(df["True"], df["Predicted"], average="weighted")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    get_accuracy_per_class(df)
    idx_names = ["Non_Damage_Score","Earthquake_Score","Fire_Score","Flood_Score"]
    scores = df[idx_names].values
    get_confusion_matrix(df["True"], df["Predicted"], label_names=label_names)
    plot_roc_curve(label_names, df["True"], scores)

if __name__ == "__main__":
    main()