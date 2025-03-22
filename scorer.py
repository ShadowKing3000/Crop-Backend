# # suitability_scorer.py
#
import numpy as np
import pandas as pd
#
# def calculate_suitability_score(input_data, crop, crop_summary):
#     """
#     Calculate a suitability score (0 to 100) for the specified crop based on input conditions.
#     Args:
#         input_data: Input features as a 1D array (e.g., [90, 42, 43, 20.879744, 75, 5.5, 220]).
#         crop: The crop label to evaluate.
#         crop_summary: DataFrame containing mean values of features for each crop.
#     Returns:
#         suitability_score: A score between 0 and 100.
#     """
#     # Get ideal conditions for the specified crop
#     ideal_conditions = crop_summary.loc[crop]
#
#     # Calculate deviation for each feature
#     deviation = np.abs(np.array(input_data) - np.array(ideal_conditions))
#
#     # Normalize each feature's deviation to a score between 0 and 100
#     # Higher deviation means lower suitability
#     feature_scores = []
#     for i in range(len(deviation)):
#         if ideal_conditions[i] == 0:
#             # If the ideal condition is 0, avoid division by zero
#             feature_score = 100 if deviation[i] == 0 else 0
#         else:
#             # Normalize deviation to a score between 0 and 100
#             feature_score = max(0, 100 - (deviation[i] / ideal_conditions[i]) * 100)
#         feature_scores.append(feature_score)
#
#     # Calculate the overall suitability score as the average of feature scores
#     suitability_score = np.mean(feature_scores)
#     return suitability_score
def calculate_suitability_score(input_data, crop, crop_summary):
    """
    Calculate a suitability score (0 to 100) for the specified crop based on input conditions.
    Args:
        input_data: Input features as a 1D array (e.g., [90, 42, 43, 20.879744, 75, 5.5, 220]).
        crop: The crop label to evaluate.
        crop_summary: DataFrame containing mean values of features for each crop.
    Returns:
        suitability_score: A score between 0 and 100.
    """
    # Get ideal conditions for the specified crop
    ideal_conditions = crop_summary.loc[crop]
    print(f"Ideal conditions for {crop}: {ideal_conditions}")

    # Calculate deviation for each feature
    deviation = np.abs(np.array(input_data) - np.array(ideal_conditions))
    print(f"Deviation from ideal conditions: {deviation}")

    # Normalize each feature's deviation to a score between 0 and 100
    # Higher deviation means lower suitability
    feature_scores = []
    for i in range(len(deviation)):
        if ideal_conditions.iloc[i] == 0:  # Use .iloc for positional indexing
            # If the ideal condition is 0, avoid division by zero
            feature_score = 100 if deviation[i] == 0 else 0
        else:
            # Normalize deviation to a score between 0 and 100
            feature_score = max(0, 100 - (deviation[i] / ideal_conditions.iloc[i]) * 100)  # Use .iloc
        feature_scores.append(feature_score)
        print(f"Feature {i} score: {feature_score}")

    # Calculate the overall suitability score as the average of feature scores
    suitability_score = np.mean(feature_scores)
    print(f"Overall suitability score for {crop}: {suitability_score:.2f}")
    return suitability_score