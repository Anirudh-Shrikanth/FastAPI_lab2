# FastAPI_lab2
Contains FastAPI_lab2 materials

This project is a mini machine learning API built with FastAPI that can predict the class of wine based on chemical features.

1. data.py
* Loads the Wine dataset from sklearn.
* Splits it into training and testing sets.
* Purpose: Prepare the data for training the model.

2. train.py
* Trains a Decision Tree classifier on the wine data.
* Saves the trained model to model/wine_model.pkl using joblib.
* Purpose: Create a model that the API can use for predictions.

3. predict.py
* Loads the saved model from disk.
* Makes predictions for new input data using the model.
* Purpose: Serve predictions whenever the API receives input.

4. main.py
* This is the FastAPI app. It defines endpoints you can call:
/	            GET	    Health check → returns {"status":"healthy"}
/predict	    POST	Accepts wine features as input and returns the predicted class (0,1,2)
/info	        GET	    Returns API info like version, author, available endpoints
/features	    GET	    Returns a list of all required input features
/reload-model	POST	Reloads the model from disk if it’s updated
