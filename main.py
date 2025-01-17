import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

data = pd.read_csv("newcsv.csv")
df = pd.DataFrame(data)
df = df.drop('Leading Cancer Sites Code', axis=1)
df = df.drop('Race Code', axis=1)
df = df.drop('States Code', axis=1)
df = df.drop('Count', axis=1)
df = df.drop('Notes', axis=1)
df = df.drop('Population', axis=1)


categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
del categorical_columns[0]
del categorical_columns[-1]
#print(categorical_columns)
encoder = OneHotEncoder(sparse=False)

# Apply one-hot encoding to the categorical columns
one_hot_encoded = encoder.fit_transform(df[categorical_columns])

#Create a DataFrame with the one-hot encoded columns
#We use get_feature_names_out() to get the column names for the encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded dataframe with the original dataframe
df_encoded = pd.concat([df, one_hot_df], axis=1)

# Drop the original categorical columns
df_encoded = df_encoded.drop(categorical_columns, axis=1)

filter = df_encoded['Age-Adjusted Rate'].str.contains("Not Applicable")
df_encoded = df_encoded[~filter]

df_encoded.to_csv("myVer", encoding='utf-8', index=False)

# Define the target variable (y)
y = df_encoded["Leading Cancer Sites"].values

# Define the features (X)
X = df_encoded.drop("Leading Cancer Sites", axis=1)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(max_depth=15,max_features=0.5,min_samples_leaf=4, min_samples_split=10,n_estimators=75)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

predictions = model.predict(X_test)
print(f"Predictions for the first 5 test samples: {predictions[:5]}")





# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame with feature names and their importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance from Random Forest Classifier')
plt.show()



# Training accuracy
train_accuracy = model.score(X_train, y_train)

# Test accuracy
test_accuracy = model.score(X_test, y_test)

# Plot the training vs test accuracy
plt.bar(['Training Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()




# change




import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Sample size to display the first few predictions
sample_size = 10  # Display the first 10 samples

# Create a DataFrame with actual and predicted values for comparison
comparison_df = pd.DataFrame({
    'Actual': y_test[:sample_size],
    'Predicted': predictions[:sample_size]
})

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit the encoder on both the actual and predicted values (combined)
all_labels = pd.concat([pd.Series(y_train), pd.Series(predictions)], axis=0)
label_encoder.fit(all_labels)

# Convert actual and predicted values to numeric
comparison_df['Actual'] = label_encoder.transform(comparison_df['Actual'])
comparison_df['Predicted'] = label_encoder.transform(comparison_df['Predicted'])

# Reverse the encoding to get the original class names for labeling
actual_labels = label_encoder.inverse_transform(comparison_df['Actual'])
predicted_labels = label_encoder.inverse_transform(comparison_df['Predicted'])

# Plotting
plt.figure(figsize=(10, 6))
# Plotting a bar chart with actual and predicted values as two bars per sample
comparison_df['Index'] = comparison_df.index
comparison_df.set_index('Index')[['Actual', 'Predicted']].plot(kind='bar', figsize=(10, 6))

# Set plot title and labels
plt.title('Actual vs Predicted for First 10 Samples')
plt.xlabel('Index')
plt.ylabel('Class Labels')
plt.xticks(ticks=range(sample_size), labels=actual_labels, rotation=45)
plt.show()