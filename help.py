import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("newcsv.csv")
df = pd.DataFrame(data)

# Drop unnecessary columns
df = df.drop(['Leading Cancer Sites Code', 'Race Code', 'States Code', 'Count', 'Notes', 'Population'], axis=1)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Apply one-hot encoding to the categorical columns
encoder = OneHotEncoder(sparse=False)
one_hot_encoded = encoder.fit_transform(df[categorical_columns])

# Create a DataFrame with the one-hot encoded columns
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded columns back to the original dataframe
df_encoded = pd.concat([df, one_hot_df], axis=1)

# Drop the original categorical columns (since they are now one-hot encoded)
df_encoded = df_encoded.drop(categorical_columns, axis=1)

# Check for missing values and drop rows if any
df_encoded = df_encoded.dropna()

# Select target columns (one-hot encoded cancer types)
target_columns = [col for col in df_encoded.columns if 'Leading Cancer Sites' in col]

# Define the target variable (y)
y = df_encoded[target_columns].values

# Define the features (X)
X = df_encoded.drop(target_columns, axis=1)

# Ensure the same number of rows in X and y
if X.shape[0] != y.shape[0]:
    print(f"Inconsistent row counts: X has {X.shape[0]} rows and y has {y.shape[0]} rows")
    raise ValueError("Row counts in X and y do not match!")

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Optionally, make predictions
predictions = model.predict(X_test)
print(f"Predictions for the first 5 test samples: {predictions[:5]}")