# # new_project.py
#
# import joblib
# import numpy as np
#
# # Load the saved model and crop summary
# model = joblib.load("crop_model1.pkl")
# crop_summary = joblib.load("crop_summary.pkl")
#
# # Define the predict_and_score function (or import it from the module if saved separately)
# def predict_and_score(model, input_data, crop_summary, desired_crop=None):
#     """
#     Predict the crop and calculate its suitability score.
#     Args:
#         model: Trained model.
#         input_data: Input features as a 2D array (e.g., [[90, 42, 43, 20.879744, 75, 5.5, 220]]).
#         crop_summary: DataFrame containing mean values of features for each crop.
#         desired_crop: The crop you want to grow (optional).
#     """
#     # If desired_crop is "wheat", treat it as "mothbeans"
#     if desired_crop and desired_crop.lower() == "wheat":
#         desired_crop = "mothbeans"
#
#     # Predict the crop
#     predicted_crop = model.predict(input_data)[0]
#
#     # If predicted crop is "mothbeans", change it to "wheat" for display
#     if predicted_crop == "mothbeans":
#         predicted_crop_display = "wheat"
#     else:
#         predicted_crop_display = predicted_crop
#     print(f"Predicted Crop: {predicted_crop_display}")
#
#     # Calculate suitability score for the predicted crop
#     predicted_score = calculate_suitability_score(input_data[0], predicted_crop, crop_summary)
#     print(f"Suitability Score for {predicted_crop_display}: {predicted_score:.2f}")
#
#     # If desired crop is provided, calculate its suitability score
#     if desired_crop:
#         desired_score = calculate_suitability_score(input_data[0], desired_crop, crop_summary)
#         if desired_crop == "mothbeans":
#           desired_crop = "wheat"
#         print(f"Suitability Score for {desired_crop}: {desired_score:.2f}")
#
# # Example usage
# new_data = [[90, 42, 43, 20.879744, 75, 5.5, 220]]
# desired_crop = "wheat"  # Specify the crop you want to grow
# predict_and_score(model, new_data, crop_summary, desired_crop=desired_crop)

# main.py
################################################################################################
# from fastapi import FastAPI, HTTPException
# import joblib
# import numpy as np
# from pydantic import BaseModel
# from scorer import calculate_suitability_score  # Import the function
#
# # Define the input data model using Pydantic
# class CropInput(BaseModel):
#     N: float
#     P: float
#     K: float
#     temperature: float
#     humidity: float
#     ph: float
#     rainfall: float
#     desired_crop: str = None  # Optional field for desired crop
#
# # Initialize FastAPI app
# app = FastAPI()
#
# # Load the saved model and crop summary
# model = joblib.load("crop_model1.pkl")
# crop_summary = joblib.load("crop_summary.pkl")
#
# # Define the prediction endpoint
# @app.post("/predict")
# def predict_crop(input_data: CropInput):
#     """
#     Predict the crop and calculate its suitability score.
#     Args:
#         input_data: Input features and optional desired crop.
#     Returns:
#         JSON response with predicted crop, suitability score, and desired crop score (if provided).
#     """
#     # Convert input data to a numpy array
#     input_array = np.array([
#         input_data.N,
#         input_data.P,
#         input_data.K,
#         input_data.temperature,
#         input_data.humidity,
#         input_data.ph,
#         input_data.rainfall
#     ]).reshape(1, -1)
#
#     # Predict the crop
#     predicted_crop = model.predict(input_array)[0]
#
#     # If predicted crop is "mothbeans", change it to "wheat" for display
#     if predicted_crop == "mothbeans":
#         predicted_crop_display = "wheat"
#     else:
#         predicted_crop_display = predicted_crop
#
#     # Calculate suitability score for the predicted crop
#     predicted_score = calculate_suitability_score(input_array[0], predicted_crop, crop_summary)
#
#     # Prepare the response
#     response = {
#         "predicted_crop": predicted_crop_display,
#         "suitability_score": round(predicted_score, 2)
#     }
#
#     # If desired crop is provided, calculate its suitability score
#     if input_data.desired_crop:
#         # If desired_crop is "wheat", treat it as "mothbeans"
#         if input_data.desired_crop.lower() == "wheat":
#             desired_crop = "mothbeans"
#         else:
#             desired_crop = input_data.desired_crop
#
#         desired_score = calculate_suitability_score(input_array[0], desired_crop, crop_summary)
#         if desired_crop == "mothbeans":
#             desired_crop = "wheat"
#         response["desired_crop"] = desired_crop
#         response["desired_crop_score"] = round(desired_score, 2)
#
#     return response
# input_data = [90, 42, 43, 20.879744, 75, 5.5, 220]
#
# # Calculate suitability score for a specific crop
# crop = "mothbeans"
# score = calculate_suitability_score(input_data, crop, crop_summary)
# print(f"Suitability Score for {crop}: {score:.2f}")
#
# # Run the FastAPI app
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# main.py

# from fastapi import FastAPI, HTTPException
# import joblib
# import numpy as np
# from pydantic import BaseModel
# from scorer import calculate_suitability_score  # Import the function
#
# # Define the input data model using Pydantic
# class CropInput(BaseModel):
#     N: float
#     P: float
#     K: float
#     temperature: float
#     humidity: float
#     ph: float
#     rainfall: float
#     desired_crop: str = None  # Optional field for desired crop
#
# # Initialize FastAPI app
# app = FastAPI()
#
# # Root route
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Crop Recommendation API!"}
#
# # Load the saved model and crop summary
# model = joblib.load("crop_model1.pkl")
# crop_summary = joblib.load("crop_summary.pkl")
#
# # Define the prediction endpoint
# @app.post("/predict")
# def predict_crop(input_data: CropInput):
#     """
#     Predict the crop and calculate its suitability score.
#     Args:
#         input_data: Input features and optional desired crop.
#     Returns:
#         JSON response with predicted crop, suitability score, and desired crop score (if provided).
#     """
#     # Convert input data to a numpy array
#     input_array = np.array([
#         input_data.N,
#         input_data.P,
#         input_data.K,
#         input_data.temperature,
#         input_data.humidity,
#         input_data.ph,
#         input_data.rainfall
#     ]).reshape(1, -1)
#
#     # Predict the crop
#     predicted_crop = model.predict(input_array)[0]
#
#     # If predicted crop is "mothbeans", change it to "wheat" for display
#     if predicted_crop == "mothbeans":
#         predicted_crop_display = "wheat"
#     else:
#         predicted_crop_display = predicted_crop
#
#     # Calculate suitability score for the predicted crop
#     predicted_score = calculate_suitability_score(input_array[0], predicted_crop, crop_summary)
#
#     # Prepare the response
#     response = {
#         "predicted_crop": predicted_crop_display,
#         "suitability_score": round(predicted_score, 2)
#     }
#
#     # If desired crop is provided, calculate its suitability score
#     if input_data.desired_crop:
#         # If desired_crop is "wheat", treat it as "mothbeans"
#         if input_data.desired_crop.lower() == "wheat":
#             desired_crop = "mothbeans"
#         else:
#             desired_crop = input_data.desired_crop
#
#         desired_score = calculate_suitability_score(input_array[0], desired_crop, crop_summary)
#         # if desired_crop == "mothbeans":
#         #     desired_crop = "wheat"
#         response["desired_crop"] = desired_crop
#         response["desired_crop_score"] = round(desired_score, 2)
#
#     return response
# input_data = [90, 42, 43, 20.879744, 75, 5.5, 220]
# crop = "wheat"
# score = calculate_suitability_score(input_data, crop, crop_summary)
# print(f"Suitability Score for {crop}: {score:.2f}")
#
# # Run the FastAPI app
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# main.py

# main.py
# main.py

from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from scorer import calculate_suitability_score  # Import the function from scorer.py

# Define the input data model using Pydantic
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    desired_crop: str = None  # Optional field for desired crop

# Initialize FastAPI app
app = FastAPI()

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Recommendation API!"}

# Load the saved model and crop summary
model = joblib.load("crop_model1.pkl")  # Use your model file name
crop_summary = joblib.load("crop_summary.pkl")

# Define the prediction endpoint
@app.post("/predict")
def predict_crop(input_data: CropInput):
    """
    Predict the crop and calculate its suitability score.
    Also calculate the suitability score for the desired crop (if provided).
    Args:
        input_data: Input features and optional desired crop.
    Returns:
        JSON response with predicted crop, suitability score, and desired crop score (if provided).
    """
    print("Received input data:", input_data)  # Debugging

    # Convert input data to a numpy array
    input_array = np.array([
        input_data.N,
        input_data.P,
        input_data.K,
        input_data.temperature,
        input_data.humidity,
        input_data.ph,
        input_data.rainfall
    ]).reshape(1, -1)
    print("Input array:", input_array)  # Debugging

    # Predict the crop
    predicted_crop = model.predict(input_array)[0]
    print("Predicted crop:", predicted_crop)  # Debugging

    # If predicted crop is "mothbeans", change it to "wheat" for display
    if predicted_crop == "mothbeans":
        predicted_crop_display = "wheat"
    else:
        predicted_crop_display = predicted_crop
    print("Predicted crop (display):", predicted_crop_display)  # Debugging

    # Calculate suitability score for the predicted crop
    try:
        predicted_score = calculate_suitability_score(input_array[0], predicted_crop, crop_summary)
        print("Predicted score:", predicted_score)  # Debugging
    except KeyError:
        print(f"Error: Crop '{predicted_crop}' not found in the dataset.")  # Debugging
        raise HTTPException(status_code=400, detail=f"Crop '{predicted_crop}' not found in the dataset.")

    # Prepare the response
    response = {
        "predicted_crop": predicted_crop_display,
        "predicted_crop_original": predicted_crop,  # Include the original predicted crop
        "predicted_crop_score": round(predicted_score, 2)
    }

    # If desired crop is provided, calculate its suitability score
    if input_data.desired_crop:
        print("Desired crop:", input_data.desired_crop)  # Debugging

        # If desired_crop is "wheat", treat it as "mothbeans"
        if input_data.desired_crop.lower() == "wheat":
            desired_crop = "mothbeans"
            desired_crop_display = "wheat"
        else:
            desired_crop = input_data.desired_crop
            desired_crop_display = desired_crop

        try:
            desired_score = calculate_suitability_score(input_array[0], desired_crop, crop_summary)
            print("Desired score:", desired_score)  # Debugging
        except KeyError:
            print(f"Error: Crop '{desired_crop}' not found in the dataset.")  # Debugging
            raise HTTPException(status_code=400, detail=f"Crop '{desired_crop}' not found in the dataset.")

        response["desired_crop"] = desired_crop_display
        response["desired_crop_score"] = round(desired_score, 2)

    print("Response:", response)  # Debugging
    return response

# Example usage of the calculate_suitability_score function
if __name__ == "__main__":
    # Your input data
    input_data = [90, 42, 43, 20.879744, 75, 5.5, 220]
    crop = "wheat"  # This will be mapped to "mothbeans" internally

    # Calculate suitability score for the desired crop
    try:
        score = calculate_suitability_score(input_data, "mothbeans", crop_summary)  # Use "mothbeans" internally
        print(f"Suitability Score for {crop}: {score:.2f}")
    except KeyError:
        print(f"Crop '{crop}' not found in the dataset.")

    # Run the FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)