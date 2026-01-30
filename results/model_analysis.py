import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_results(results):
    predicted = results["Predicted"]
    actual = results["Actual"]

    predicted_labels = []
    actual_labels = []

    labels = ["Non_Damage", "Land_Disaster", "Fire_Disaster", "Water_Disaster"]

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
    for label in ["Non_Damage", "Land_Disaster", "Fire_Disaster", "Water_Disaster"]:
        total = results[results["True"] == label]
        correct = total[total["Correct"] == True]
        acc = len(correct) / len(total)
        print(f"{label}\t{acc}")

    non_damage = results[results["True"] == "Non_Damage"]
    n_normal = len(non_damage[non_damage["Predicted"] == "Non_Damage"])
    n_fire = len(non_damage[non_damage["Predicted"] == "Fire_Disaster"])
    n_flood = len(non_damage[non_damage["Predicted"] == "Water_Disaster"])
    n_land = len(non_damage[non_damage["Predicted"] == "Land_Disaster"])

    print(f"Normal: {n_normal}\t Fire: {n_fire}\t Flood: {n_flood}\t Land: {n_land}")


def calculate_precision(results):
    """
                                            True
                            | Normal | Earthquake | Fire | Flood
                            --------------------------------------
                    Normal |
    Pred        Earthquake |
                      Fire |
                     Flood |
    """
    true_normal = len(results[results["True"] == "Non_Damage"])
    true_earthquake = len(results[results["True"] == "Land_Disaster"])
    true_fire = len(results[results["True"] == "Fire_Disaster"])
    true_flood = len(results[results["True"] == "Water_Disaster"])

    print(f"{true_normal=}\t{true_earthquake=}\t{true_fire=}\t{true_flood=}")

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
    # results = pd.read_csv("results_updated.csv")
    # results["Correct"] = results["True"] == results["Predicted"]
    # num_correct = len(results[results["Correct"] == True])
    # print(f"Accuracy: {round(num_correct / len(results), 3)}")
    label_names = ["Non_Damage", "Land_Disaster", "Fire_Disaster", "Water_Disaster"]

    results = pd.read_csv("resnet50_results.csv")
    df = get_results(results)
    # calculate_precision(results)
    get_accuracy_per_class(df)
    get_confusion_matrix(df["True"], df["Predicted"], label_names=label_names)

if __name__ == "__main__":
    main()