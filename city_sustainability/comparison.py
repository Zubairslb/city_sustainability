from city_sustainability.quality import life_quality
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
    class_percentages_act = {
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

def qol_comparison(y_pred, y_test):

    # Initialize the count lists for preductions
    High_quality_pred = []
    Med_quality_pred = []
    Low_quality_pred = []
    quality_pred = []

    # Loop over all the images
    for image in y_pred:
        # Get the classification for the current image
        class_percentages_pred, sorted_metrics_pred, classification_pred = life_quality(image)
        quality_pred.append(classification_pred)
        # Update the count lists based on the classification
        if classification_pred == 'High quality of life':
            High_quality_pred.append(1)
        elif classification_pred == 'Medium quality of life':
            Med_quality_pred.append(1)
        elif classification_pred == 'Low quality of life':
            Low_quality_pred.append(1)

    # Get the total counts for each classification
    total_high_quality_pred = len(High_quality_pred)
    total_med_quality_pred = len(Med_quality_pred)
    total_low_quality_pred = len(Low_quality_pred)

    # Initialize the count lists for actual
    High_quality_act = []
    Med_quality_act = []
    Low_quality_act = []
    quality_act = []

    # Loop over all the images
    for image in y_test:
        # Get the classification for the current image
        class_percentages_act, sorted_metrics_act, classification_act = life_quality(image)
        quality_act.append(classification_act)
        # Update the count lists based on the classification
        if classification_act == 'High quality of life':
            High_quality_act.append(1)
        elif classification_act == 'Medium quality of life':
            Med_quality_act.append(1)
        elif classification_act == 'Low quality of life':
            Low_quality_act.append(1)

    # Get the total counts for each classification
    total_high_quality_act = len(High_quality_act)
    total_med_quality_act = len(Med_quality_act)
    total_low_quality_act = len(Low_quality_act)

    # Data
    labels = ['High', 'Medium', 'Low']
    actual = [total_high_quality_act, total_med_quality_act, total_low_quality_act]
    prediction = [total_high_quality_pred, total_med_quality_pred, total_low_quality_pred]

    # Bar plot settings
    bar_width = 0.35
    index = np.arange(len(labels))

    # Create the bar plots
    plt.bar(index, actual, bar_width, label='Actual')
    plt.bar(index + bar_width, prediction, bar_width, label='Prediction')

    # Set labels and title
    plt.xlabel('Quality of Life')
    plt.ylabel('Count')
    plt.title('Comparison of Actual and Prediction')

    # Set x-axis ticks and labels
    plt.xticks(index + bar_width / 2, labels)

    # Add legend on the right
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Create the dataframe
    data = {'Quality of Life': labels, 'Actual': actual, 'Prediction': prediction}
    df = pd.DataFrame(data)

    # Calculate confusion matrix
    cm = confusion_matrix(quality_act, quality_pred, labels=labels)

    # Plot confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    return df, plt.show()
