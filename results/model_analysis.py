import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_results(results):
    predicted = results["Predicted"]
    actual = results["True"]

    predicted_labels = []
    actual_labels = []

    labels = ["Non-Damage", "Earthquake", "Fire", "Flood"]

    for x, y in zip(predicted, actual):
        predicted_labels.append(labels[int(x)])
        actual_labels.append(labels[int(y)])

    df = pd.DataFrame({
        "Predicted": predicted_labels,
        "True": actual_labels
    })
    df["Correct"] = df["Predicted"] == df["True"]

    return df

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

def plot_precision_recall_curve(y_true, y_pred, label_names):
    precision = {}
    recall = {}
    for i in range(len(label_names)):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        plt.plot(recall[i], precision[i], lw=2, label="class {}".format(label_names[i]))

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title("Precision v. Recall Curve")
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

    results = pd.read_csv("resnet50_results.csv")
    df = get_results(results)
    precision = calculate_precision(df["True"], df["Predicted"], average="weighted")
    recall = calculate_recall(df["True"], df["Predicted"], average="weighted")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    get_accuracy_per_class(df)
    # get_confusion_matrix(df["True"], df["Predicted"], label_names=label_names)
    # plot_precision_recall_curve(df["True"], df["Predicted"], label_names)

if __name__ == "__main__":
    main()