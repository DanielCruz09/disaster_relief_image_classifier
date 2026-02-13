import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Natural Disaster Image Classification")

st.set_page_config(
    page_title="Natural Disaster Classification Model Analytics",
    page_icon=":rescue_worker_helmet:",
    layout="wide"
)

st.markdown(
    "### Welcome! This is a page dedicated to demonstrating how a classification model can " \
    "classify natural disaster images to help inform rescue workers about natural disaster cases. " \
    "Please see below to check the model's performance."
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

data_load_state = st.text("Loading data...")
data = load_data("resnet50_results.csv")
data_load_state.text("Loading data...done!")

disaster = st.selectbox(
    "Choose a disaster",
    data["True"].unique(),
    index=None,
    placeholder="Choose a natural disaster"
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Accuracy per Class")
    get_accuracy_per_class(data, disaster)

with col2:
    st.subheader("Count per Class")
    get_counts(data, disaster)