import numpy as np

def life_quality(array):

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

    class_sums = np.sum(array, axis=(0, 1))
    total_sum = np.sum(class_sums)
    class_percentages = {Class_labels[i]: round((sum / total_sum), 2) for i, sum in enumerate(class_sums)}

    # Calculate metric score
    environmental_metric = class_percentages.get('Rangeland', 0) + class_percentages.get('Tree', 0) + class_percentages.get('Water', 0)
    infrastructure_metric = class_percentages.get('Building', 0) + class_percentages.get('Developed space', 0) + class_percentages.get('Road', 0)
    land_metric = class_percentages.get('Bareland', 0) + class_percentages.get('Other', 0) + class_percentages.get('Agriculture land', 0)

    # Create a dictionary to hold the metric scores
    metrics = {
        'Environmental': environmental_metric,
        'Infrastructure': infrastructure_metric,
        'Land': land_metric
    }

    # Sort the metric scores in descending order
    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)

    # Classify the quality of life based on the sorted metrics
    classification = ''
    if sorted_metrics == [('Environmental', environmental_metric), ('Infrastructure', infrastructure_metric), ('Land', land_metric)] or sorted_metrics == [('Infrastructure', infrastructure_metric), ('Environmental', environmental_metric), ('Land', land_metric)]:
        classification = 'High quality of life'
    elif sorted_metrics == [('Environmental', environmental_metric), ('Land', land_metric), ('Infrastructure', infrastructure_metric)] or sorted_metrics == [('Infrastructure', infrastructure_metric), ('Land', land_metric), ('Environmental', environmental_metric)] or sorted_metrics == [('Land', land_metric), ('Infrastructure', infrastructure_metric), ('Environmental', environmental_metric)]:
        classification = 'Medium quality of life'
    elif sorted_metrics == [('Land', land_metric), ('Environmental', environmental_metric), ('Infrastructure', infrastructure_metric)]:
        classification = 'Low quality of life'

    return class_percentages, sorted_metrics, classification