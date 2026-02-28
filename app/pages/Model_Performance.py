import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.preprocessing import label_binarize
from models.resnet50 import ResNet50
import torch
import numpy as np

st.title("Natural Disaster Image Classification")

st.set_page_config(
    page_title="Natural Disaster Classification Model Analytics",
    page_icon=":rescue_worker_helmet:",
    layout="wide"
)

classes_to_colors = {
    "Earthquake": "tab:orange",
    "Fire": "tab:red",
    "Non-Damage": "tab:green",
    "Flood": "tab:blue"
}

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["Correct"] = df["Predicted"] == df["True"]
    return df

def get_counts(results, option=None):
    if option:
        classes = [option]
        counts = [
            len(results[results["True"] == label]) for label in classes
        ]
        predictions = [
            len(results[results["Predicted"] == label]) for label in classes
        ]
        data = {
            "True": counts,
            "Predicted": predictions
        }
        x = np.arange(len(classes))
        width = 0.25
        multiplier = 0

        fig, ax = plt.subplots()
        for k, v in data.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, v, width, label=k)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        ax.set_title("Count per Class")
        ax.set_ylabel("Count")
        ax.set_xlabel("Natural Disaster")
        ax.set_xticks(x + width, classes)
        ax.legend(loc="upper left", ncols=3)
        st.pyplot(fig)
        return

    classes = classes_to_colors.keys()
    counts = [
        len(results[results["True"] == label]) for label in classes
    ]
    predictions = [
        len(results[results["Predicted"] == label]) for label in classes
    ]
    data = {
        "True": counts,
        "Predicted": predictions
    }
    x = np.arange(len(classes))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots()
    for k, v in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, v, width, label=k)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_title("Count per Class")
    ax.set_ylabel("Count")
    ax.set_xlabel("Natural Disaster")
    ax.set_xticks(x + width, classes)
    ax.legend(loc="upper left", ncols=3)
    st.pyplot(fig)

def get_accuracy_per_class(results, option=None):
    accuracies = {}
    if option:
        total = results[results["True"] == option]
        correct = total[total["Correct"] == True]
        acc = len(correct) / len(total)
        accuracies[option] = acc
        bar_colors = classes_to_colors.values()
        fig, ax = plt.subplots()
        bar_container = ax.bar(accuracies.keys(), accuracies.values(), color=classes_to_colors[option])
        ax.set(ylabel="Accuracy", xlabel="Natural Disaster", title=f"Accuracy for Class: {option}")
        ax.bar_label(bar_container, fmt="{:,.3f}")
        st.pyplot(fig)
        return
    
    for label in classes_to_colors.keys():
        total = results[results["True"] == label]
        correct = total[total["Correct"] == True]
        acc = len(correct) / len(total)
        accuracies[label] = acc
    
    bar_colors = classes_to_colors.values()
    fig, ax = plt.subplots()
    bar_container = ax.bar(accuracies.keys(), accuracies.values(), color=bar_colors)
    ax.set(ylabel="Accuracy", xlabel="Natural Disaster", title="Accuracy per Class")
    ax.bar_label(bar_container, fmt="{:,.3f}")
    st.pyplot(fig)

def get_precision_recall_f1(results):
    precision_values = precision_score(results["True"], results["Predicted"], average="weighted")
    recall_values = recall_score(results["True"], results["Predicted"], average="weighted")
    f1_values = f1_score(results["True"], results["Predicted"], average="weighted")

    fig, ax = plt.subplots()
    bar_container = ax.bar(
        ["Precision", "Recall", "F1"], 
        [precision_values, recall_values, f1_values],
        color=["tab:blue", "tab:orange", "tab:green"]
    )
    ax.set(title="Precision, Recall, F1", ylabel="Score", xlabel="Metric")
    ax.bar_label(bar_container, fmt="{:,.3f}")
    st.pyplot(fig)

def draw_confusion_matrix(y_true, y_pred, label_names):

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
    st.pyplot(fig)

def draw_roc_curve(label_names, y_true, y_score):
    # See https://vitalflux.com/roc-curve-auc-python-false-positive-true-positive-rate/ for reference
    y_true_bin = label_binarize(y_true, classes=label_names)
    fig = plt.figure(figsize=(10, 6))

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
    st.pyplot(fig)

def get_model(weights_path=None):
    resnet50 = ResNet50(num_classes=4, lr=0.001)
    weights = torch.load(weights_path, map_location="cpu")
    resnet50.model.load_state_dict(weights["model_state_dict"])
    return weights, resnet50

def draw_loss_history(checkpoint):
    epochs = checkpoint["epochs"]
    loss_history = checkpoint["loss"]
    fig, ax = plt.subplots()
    ax.plot(epochs, loss_history, "bo--")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    st.pyplot(fig)

st.divider()

data_load_state = st.text("Loading data...")
data = load_data("results/resnet50_results.csv")
data_load_state.text("Loading data...done!")

disaster = st.selectbox(
    "Choose a disaster",
    data["True"].unique(),
    index=None,
    placeholder="Choose a natural disaster"
)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Accuracy per Class")
    get_accuracy_per_class(data, disaster)

with col2:
    st.subheader("Count per Class")
    get_counts(data, disaster)

with col3:
    st.subheader("Precision, Recall, F1")
    get_precision_recall_f1(data)

st.divider()

draw_confusion_matrix(data["True"], data["Predicted"], ["Earthquake", "Fire", "Non-Damage", "Flood"])
st.divider()

idx_names = ["Earthquake_Score","Fire_Score","Non_Damage_Score","Flood_Score"]
scores = data[idx_names].values
st.subheader("ROC Curve for each Class")
draw_roc_curve(["Earthquake", "Fire", "Non-Damage", "Flood"], data["True"], scores)

st.divider()
st.subheader("Loss during Training")
checkpoint, model = get_model("models/model_weights.pth")
draw_loss_history(checkpoint)