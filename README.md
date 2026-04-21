Hotel Cancellation Predictor

In this project, we build; a machine learning system that helps to predict whether a hotel booking is likely to be cancelled,using booking details and other details like guest information. The application is developed with the help of scikit-learn for modelling and Streamlit for deployment.

Project Overview:

The project has a  complete machine learning workflow, which starts from  data preparation and feature engineering  and through to model training, evaluation, tuning, and deployment in a very user-friendly web interface.

We compared: Logistic Regression, which is the baseline , Decision Tree, and Random Forest. The tuned Random Forest is selected as the final model for deployment.

Key modelling choices:


The feature we removed was the arrival year, as the dataset only covers 2017–2018 and adds no general value.
2. Class imbalance is nicely handled with the help of balanced weighting across all models.
3. Predictions are based on probabilities with a nice 0.4 threshold, improving the detection of cancellations
4. Two additional features were also created, which are total nights and total guests.

 Interpretability features: 

The application contains several interpretability features which can help users to understand predictions, like a feature importance chart highlighting key factors, a key drivers panel showing what influences each result, a plain-English explanation generated from user inputs, and also a very nicely made summary table displaying all inputs with their exact impact on the prediction. The repository contains the main Streamlit application, which is app.py, a full pipeline script that is made as a full code and a step-by-step notebook named steps, a requirements file, and the dataset, which I used from Kaggle.

https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset

Application Features:

The application has a prediction output that displays the risk level with the associated probability. It also includes a key drivers panel that helps to highlight the main factors influencing the result. A feature importance chart shows the most significant variables, while an explanation text describes the prediction in simple, easy-to-understand language. In addition, a summary table lists all input values and their impact on the prediction.

Target variable:

The target variable indicates whether a booking is cancelled. A value of 1 indicates that the reservation was cancelled, while a value of 0 means the booking was completed successfully.

Features overview:

The model uses a range of information, which includes guest details such as the number of adults and children, and also previous bookings; stay details such as the number of nights, room type, and meal plan; booking behaviour, including lead time and booking channel; as well as pricing and special requests. In the application, these inputs are displayed using simple, user-friendly labels, while the original dataset values are processed internally to maintain model consistency.

Running the Full Script:

By using python ‘’ full_model_code.py ‘’, the full pipeline evaluates models, performs tuning, and prints results.

Limitations:

The project has a very few important limitations. The dataset covers only two years and comes from a single hotel group, which may limit the model's generalizability to other settings. Its performance may also change over time if booking behaviour shifts. For this reason, the predictions should be used to support decision-making rather than replace human judgment.
