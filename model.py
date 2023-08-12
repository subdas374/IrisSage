# Step 1: Import libraries
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score


iris_df = pd.read_csv("./Iris_data.csv")
iris_df.drop("Id", axis=1, inplace=True)


X = iris_df.drop("Species",axis=1)
y = iris_df["Species"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 4: Choose a machine learning algorithm (SVM in this case)
model = SVC()


# Step 5: Train the model on the training data
model.fit(X_train, y_train)





# # Step 6: Evaluate the model on the testing data
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print("Accuracy:", accuracy)

# # Step 7: Make predictions using the trained model (optional)
# new_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example new data
# predicted_class = model.predict(new_data)
# print("Predicted class for new data:", predicted_class)

import pickle
pickle.dump(model,open("model.pkl", "wb"))