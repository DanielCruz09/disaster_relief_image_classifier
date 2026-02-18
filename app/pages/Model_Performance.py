import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize

st.title("Natural Disaster Image Classification")

st.set_page_config(
    page_title="Natural Disaster Classification Model Analytics",
    page_icon=":rescue_worker_helmet:",
    layout="wide"
)

classes_to_colors = {
    "Earthquake": "tab:red",
    "Fire": "tab:blue",
    "Non-Damage": "tab:green",
    "Flood": "tab:orange"
}

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["Correct"] = df["Predicted"] == df["True"]
    return df

def get_counts(results, option=None):
    if option:
        fig, ax = plt.subplots()
        bar_container = ax.bar(option, len(results[results["True"] == option]), color=classes_to_colors[option])
        ax.set(ylabel="Count", xlabel="Natural Disaster", title=f"Count for Class: {option}")
        ax.bar_label(bar_container, fmt="{:,.0f}")
        st.pyplot(fig)
        return

    classes = results["True"].unique()
    counts = results["True"].value_counts()
    fig, ax = plt.subplots()
    bar_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    bar_container = ax.bar(classes, counts, color=bar_colors)
    ax.set(ylabel="Count", xlabel="Natural Disaster", title="Count per Class")
    ax.bar_label(bar_container, fmt="{:,.0f}")
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

st.divider()

data_load_state = st.text("Loading data...")
data = load_data("resnet50_results.csv")
data_load_state.text("Loading data...done!")

disaster = st.selectbox(
    "Choose a disaster",
    data["True"].unique(),
    index=None,
    placeholder="Choose a natural disaster"
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Accuracy per Class")
    get_accuracy_per_class(data, disaster)

with col2:
    st.subheader("Count per Class")
    get_counts(data, disaster)

st.divider()

draw_confusion_matrix(data["True"], data["Predicted"], ["Earthquake", "Fire", "Non-Damage", "Flood"])
st.divider()

idx_names = ["Earthquake_Score","Fire_Score","Non_Damage_Score","Flood_Score"]
scores = data[idx_names].values
st.subheader("ROC Curve for each Class")
draw_roc_curve(["Earthquake", "Fire", "Non-Damage", "Flood"], data["True"], scores)