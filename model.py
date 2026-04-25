import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load cleaned Tetouan dataset
df = pd.read_csv("tetdata.csv")

# Convert DateTime column into useful time features
df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True, errors="coerce")
df["hour"] = df["DateTime"].dt.hour
df["day"] = df["DateTime"].dt.day
df["month"] = df["DateTime"].dt.month
df["dayofweek"] = df["DateTime"].dt.dayofweek

# Target variable
target = "Zone 1 Power Consumption"

# Features used for prediction
features = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "general diffuse flows",
    "diffuse flows",
    "hour",
    "day",
    "month",
    "dayofweek"
]

# Remove any invalid rows after DateTime conversion
df = df.dropna()

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Save metrics to text file
with open("metrics.txt", "w") as f:
    f.write("Tetouan City Power Consumption Prediction\n")
    f.write("Target Variable: Zone 1 Power Consumption\n")
    f.write("Model: Linear Regression\n")
    f.write("=" * 50 + "\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.2f}\n")
    f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Zone 1 Power Consumption")
plt.ylabel("Predicted Zone 1 Power Consumption")
plt.title("Tetouan Zone 1 Power Consumption: Actual vs Predicted")
plt.grid(True, alpha=0.3)

# Add diagonal reference line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.tight_layout()
plt.savefig("model_results.png", dpi=120)

print("Model training completed successfully.")
print("model_results.png and metrics.txt have been generated.")