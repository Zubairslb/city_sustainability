from city_sustainability.quality import life_quality
import numpy as np 
import pandas as pd

def class_comparison(y_pred, y_test):
    
    Class_labels = {
    0: "Other",
    1: "Bareland",
    2: "Rangeland",
    3: "Developed space",
    4: "Road",
    5: "Tree",
    6: "Water",
    7: "Agriculture land",
    8: "Building"
    }

    # Dictionary of the class distribution in all the prediction labels

    class_sums = np.sum(y_pred, axis=(0, 1, 2))
    total_sum = np.sum(class_sums)
    class_percentages_pred = {
        Class_labels[i]: round((class_sum / total_sum) * 100, 2) for i, class_sum in enumerate(class_sums)
        }

    # Dictionary of the class distribution in all the actual labels

    class_sums = np.sum(y_test, axis=(0, 1, 2))
    total_sum = np.sum(class_sums)
    class_percentages_pred = {
        Class_labels[i]: round((class_sum / total_sum) * 100, 2) for i, class_sum in enumerate(class_sums)
        }

    df = pd.DataFrame({
    "Actual": class_percentages_act,
    "Predicted": class_percentages_pred
    })
    df['Difference'] = round(df['Actual'] - df['Predicted'],2)
    df = df.sort_values('Difference', ascending=False)

    df2 = df[["Actual","Predicted"]]
    df2.plot.bar()
    plt.xlabel("Classes")
    plt.ylabel("Percentage")
    plt.title("Actual vs Predicted Class Percentages")
    plt.legend()

    return df, plt.show()