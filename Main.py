import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("data/sample_vehicle_data.csv")

X = data[['Wind Shield Length', 'Front Angle', 'Rear Window Length', 'Back Angle', 'Roof Length']]
y = data['Drag Coefficient']
# Check for missing values
data.isnull().sum()

# Fill or drop missing values (if any)
data = data.dropna()

# Split data into features and target


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot to visualize relationships
sns.pairplot(data)
plt.show()

# Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Function to train and evaluate models
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
        print(f"{name}: MSE = {mse}, R2 = {r2}")
    return results

# Train and evaluate
results = train_and_evaluate(models, X_train_scaled, X_test_scaled, y_train, y_test)


import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Test MAE: {mae}")

# Predict and calculate R2 score
y_pred_nn = model.predict(X_test_scaled).flatten()
r2_nn = r2_score(y_test, y_pred_nn)
print(f"Test R2: {r2_nn}")

# Append TensorFlow model results
results['Neural Network'] = {'MSE': mean_squared_error(y_test, y_pred_nn), 'R2': r2_nn}

# Display results
for model_name, metrics in results.items():
    print(f"{model_name}: MSE = {metrics['MSE']}, R2 = {metrics['R2']}")
