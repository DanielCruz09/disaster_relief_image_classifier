import pandas as pd

def get_accuracy_per_class(results):
    for label in ["Non_Damage", "Land_Disaster", "Fire_Disaster", "Water_Disaster"]:
        total = results[results["True"] == label]
        correct = total[total["Correct"] == True]
        acc = len(correct) / len(total)
        print(f"{label}\t{acc}")

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

def main():
    results = pd.read_csv("results.csv")
    results["Correct"] = results["True"] == results["Predicted"]
    num_correct = len(results[results["Correct"] == True])
    print(f"Accuracy: {round(num_correct / len(results), 2)}")

    calculate_precision(results)
    get_accuracy_per_class(results)

if __name__ == "__main__":
    main()