import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Natural Disaster Image Classification")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["Correct"] = df["Predicted"] == df["True"]
    return df

def get_accuracy_per_class(results, option=None):
    accuracies = {}
    if option:
        total = results[results["True"] == option]
        correct = total[total["Correct"] == True]
        acc = len(correct) / len(total)
        accuracies[option] = acc
        bar_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
        fig, ax = plt.subplots()
        bar_container = ax.bar(accuracies.keys(), accuracies.values(), color=bar_colors)
        ax.set(ylabel="Accuracy", xlabel="Natural Disaster", title="Accuracy per Class")
        ax.bar_label(bar_container, fmt="{:,.3f}")
        st.pyplot(fig)
        return
    
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
    st.pyplot(fig)

data_load_state = st.text("Loading data...")
data = load_data("resnet50_results.csv")
data_load_state.text("Loading data...done!")

# st.subheader("Raw Data")
disaster = st.selectbox(
    "Choose a disaster",
    data["True"].unique(),
    index=None,
    placeholder="Choose a natural disaster"
)
# st.write(data[data["True"] == disaster])

st.subheader("Accuracy per Class")
get_accuracy_per_class(data, disaster)