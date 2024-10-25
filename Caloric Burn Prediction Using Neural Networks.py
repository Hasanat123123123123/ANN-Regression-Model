import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Applying seed
np.random.seed(42)


# Loading dataset
df = pd.read_csv('NumpyRegCSV_Data.csv')
print(df.head())

# # Checking for null values
# print(df.isnull().sum())


# Filling null values
mean_calories = df['Calories'].mean()
df['Calories'] = df['Calories'].fillna(mean_calories)  # Directly assign the filled column back to df

# Converting columns to numpy arrays
x = df['Duration'].values.reshape(-1, 1)  # Duration as independent variable
y = df['Calories'].values.reshape(-1, 1)  # Calories as dependent variable


# # Initialize scalers for individual column Durarion
# scaler_duration = StandardScaler()
# x = scaler_duration.fit_transform(x)

np.random.seed(10)

# Number of data points
N = x.shape[0]

# Shuffle indices for splitting the data
idx = np.arange(N)
np.random.shuffle(idx)

# Splitting indices for train (80%) and validation (20%)
train_size = int(0.8 * N)
idx_train = idx[:train_size]
idx_test = idx[train_size:]

# Splitting data using shuffled indices
x_train, y_train = x[idx_train], y[idx_train]
x_val, y_val = x[idx_test], y[idx_test]


# Plotting training data
plt.figure('1')
plt.scatter(x_train, y_train)
plt.xlabel('Duration')
plt.ylabel('Calories')
plt.title(f'Duration vs Calories (Training Data)')
plt.show()

# Plotting validation data
plt.figure('2')
plt.scatter(x_val, y_val, color='m')
plt.xlabel('Duration (x_val)')
plt.ylabel('Calories (y_val)')
plt.title(f'Duration vs Calories (Validation Data)')
plt.show()

# Training loop
# Initialize parameters
trainLosses = []
valLosses = []
lr = 0.001 # Learning rate
w = np.random.randn(1)*0.1 # Initialize weight randomly
b = np.random.randn(1)*0.1  # Initialize bias randomly

# Training loop for 100 epochs
for i in range(99):
    # Forward pass (Prediction)
    yhat = w * x_train + b  # Predicted calories from duration
    
    # MSE loss calculation
    error = yhat - y_train
    loss = (error ** 2).mean()  # Mean Squared Error
    trainLosses.append(loss)  # Save train loss
    
    # Compute gradients for weight and bias
    db = 2 * error.mean()  # Gradient of bias
    dw = 2 * (x_train * error).mean()  # Gradient of weight
    
    # Update weight and bias using gradient descent
    b = b - lr * db
    w = w - lr * dw

    # Validation prediction and MSE loss
    yhatVal = w * x_val + b
    errorVal = yhatVal - y_val
    valLoss = (errorVal ** 2).mean()  # Mean Squared Error for validation
    valLosses.append(valLoss)

    # Stopping condition if the validation loss is small enough
    if valLoss < 0.0001:
        break
    
    # Print progress
    print(f'Epoch {i}: train loss={loss}, val loss={valLoss}, w={w}, b={b}')


    # Training data plot
    plt.figure('3')
    plt.cla()  # Clear the current axis
    plt.scatter(x_train, y_train)
    plt.scatter(x_train, yhat)
    plt.title(f'Epoch={i}, Loss={loss}, w={w}, b={b}')
    plt.show(block=False)
    plt.pause(1)

    # Validation data plot
    plt.figure('4')
    plt.cla()  # Clear the current axis
    plt.scatter(x_val, y_val)
    plt.scatter(x_val, yhatVal)
    plt.title(f'Epoch={i}, ValLoss={valLoss}, w={w}, b={b}')
    plt.show(block=False)
    plt.pause(1)

    # Train Loss vs Epoch plot
    plt.figure('5')
    plt.cla()  # Clear the current axis
    plt.plot(trainLosses)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss vs Epoch')
    plt.show(block=False)
    plt.pause(1)

    # Validation Loss vs Epoch plot
    plt.figure('6')
    plt.cla()  # Clear the current axis
    plt.plot(valLosses, color='m')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Epoch')
    plt.show(block=False)
    plt.pause(1)

# Ensure the last plots remain visible
plt.show(block=True)



