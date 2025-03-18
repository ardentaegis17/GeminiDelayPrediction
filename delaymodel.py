## GOAL: Build a RandomForest Classifier that predicts the category of delay

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
import joblib

TODAY = pd.Timestamp.today().normalize()
MODE = "TEST"  ## TRAIN a new RF model or TEST a saved RF model
raw_data = "Raw_Data_12032025"

df = pd.read_excel("GeminiTracker Analysis.xlsx", sheet_name="Training Data", header = 0)

# check for missing values
print(df.isnull().sum())
# Fill missing values (example: fill with median)
# df.fillna(df.median(),inplace=True)

# convert dates using pd.to_datetime()
df["Prev ETB"] =pd.to_datetime(df["Prev ETB"])
df["CST ETB"] = pd.to_datetime(df["CST ETB"])
df["Date Predicted"] = pd.to_datetime(df["Date Predicted"])

# Extract the Week, Day and Hour from Prev and CST ETB
df["Prev Week"] = df["Prev ETB"].dt.isocalendar().week
df["Prev Day"] = df["Prev ETB"].dt.dayofweek
df["Prev Hour"] = df["Prev ETB"].dt.hour

df["CST Week"] = df["CST ETB"].dt.isocalendar().week
df["CST Day"] = df["CST ETB"].dt.dayofweek
df["CST Hour"] = df["CST ETB"].dt.hour

# Obtain Prev Delay by subtracting Actual Prev Delay from Fcst Prev Delay
df["Prev Delay"] = df["Actual Prev Delay"] - df["Fcst Prev Delay"]

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['SSY','Prev. Port', 'Port', 'LEG'], drop_first=True)
print(df.columns)

#Define Features (X) and Target (y)
X = df.drop(columns=["Prev ETB", "VSL NAME", "VOY NO", "CST ETB", "Fcst Prev Delay", "Actual Prev Delay","Weather Multiplier","Total Time Needed",
                     "Predicted Delay (HOURS)","Total Delay","Date Predicted","Actual Delay","Actual Time Needed",
                     "Difference","Total Delay Category","Actual Delay Category","Correct?","Comments"])

y = df["Actual Delay Category"]

if MODE == "TRAIN":
    #Encode the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print(label_encoder.classes_)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, labels = [0,1,2,3]))

    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)
    grid_search.fit(X_train, y_train)

    # Best parameters
    print("Best Parameters:", grid_search.best_params_)

    # Train the model with the best parameters
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Get feature importances
    importances = best_model.feature_importances_
    feature_names = X.columns

    # Plot feature importance
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.show()


    # Save the model
    joblib.dump(best_model, f'vessel_delay_category_RFM_{TODAY.strftime('%Y-%m-%d')}.pkl')

    # Save the label encoder (to decode predictions later)
    joblib.dump(label_encoder, f'label_encoder_{TODAY.strftime('%Y-%m-%d')}.pkl')

if MODE == "TEST":

    pred_sheet = "Predicted_Delays_" + raw_data[-8:]

    # Load the model
    loaded_model = joblib.load('vessel_delay_category_RFM_13032025.pkl')

    # Obtain number of trees in Random Forest
    print(f"Number of trees: {len(loaded_model.estimators_)}")

    # Load the label encoder
    loaded_label_encoder = joblib.load('label_encoder_13032025.pkl')

   # Visualize the first 3 trees
    for i, estimator in enumerate(loaded_model.estimators_[:3]):
        plt.figure(figsize=(20, 10))
        plot_tree(estimator, filled=True, proportion=True, feature_names=X.columns,class_names=loaded_label_encoder.classes_)
        plt.title(f"Tree {i+1}")
        plt.show()

    # New data (replace with your new data)
    new_data = pd.read_excel("GeminiTracker Analysis.xlsx", sheet_name=raw_data, header = 0)

    key = new_data[["SSY", "VOY NO", "VSL NAME", "LEG"]]

    # Ensure the new data has the same features as the training data
    new_data = pd.get_dummies(new_data)
    new_data = new_data.reindex(columns=X.columns, fill_value=0)

    # Make predictions
    predicted_category = loaded_model.predict(new_data)

    # Decode the predicted category
    predicted_category_label = loaded_label_encoder.inverse_transform(predicted_category)
    new_data["Predicted Delay Category"] = predicted_category_label

    new_data = pd.concat([key,new_data],axis = 1)

    print(new_data.head())

    
    with pd.ExcelWriter("GeminiTracker Analysis.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    # Write the DataFrame to a new or existing sheet
        new_data.to_excel(writer, sheet_name=pred_sheet, index=False)
        print(f"DataFrame written to {pred_sheet}' sheet in the existing workbook.")
