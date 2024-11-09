<<<<<<< HEAD
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("Jobs_NYC_Postings.csv")

# Columns to drop
drop_columns = [
    'Job ID',            
    'Agency',            
    'Posting Type',      
    'Division/Work Unit',
    'Additional Information', 
    'To Apply',          
    'Recruitment Contact', 
    'Posting Date',      
   'Post Until',        
    'Posting Updated',   
    'Process Date'       
]

# Drop 
data = df.drop(columns=drop_columns, errors='ignore')

print("Columns after dropping unnecessary ones:", data.columns)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Job Category'] = label_encoder.fit_transform(df['Job Category'])
df['Career Level'] = label_encoder.fit_transform(df['Career Level'])
df['Full-Time/Part-Time indicator'] = label_encoder.fit_transform(df['Full-Time/Part-Time indicator'])

data['Salary Midpoint'] = (data['Salary Range From'] + data['Salary Range To']) / 2
X = pd.get_dummies(data, columns=['Job Category', 'Full-Time/Part-Time indicator', 'Career Level'], drop_first=True)

# Feature selection and target variable

X = pd.get_dummies(X, drop_first=True) # Drop target variable from features
y = data['Salary Midpoint']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the RandomForest Regressor model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Save the model
dump(model, "Salary_Prediction_Model.joblib", compress=2)
=======
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("Jobs_NYC_Postings.csv")

# Columns to drop
drop_columns = [
    'Job ID',            
    'Agency',            
    'Posting Type',      
    'Division/Work Unit',
    'Additional Information', 
    'To Apply',          
    'Recruitment Contact', 
    'Posting Date',      
   'Post Until',        
    'Posting Updated',   
    'Process Date'       
]

# Drop 
data = df.drop(columns=drop_columns, errors='ignore')

print("Columns after dropping unnecessary ones:", data.columns)

# Encode categorical variables
label_encoder = LabelEncoder()
df['Job Category'] = label_encoder.fit_transform(df['Job Category'])
df['Career Level'] = label_encoder.fit_transform(df['Career Level'])
df['Full-Time/Part-Time indicator'] = label_encoder.fit_transform(df['Full-Time/Part-Time indicator'])

data['Salary Midpoint'] = (data['Salary Range From'] + data['Salary Range To']) / 2
X = pd.get_dummies(data, columns=['Job Category', 'Full-Time/Part-Time indicator', 'Career Level'], drop_first=True)

# Feature selection and target variable

X = pd.get_dummies(X, drop_first=True) # Drop target variable from features
y = data['Salary Midpoint']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the RandomForest Regressor model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Save the model
dump(model, "Salary_Prediction_Model.joblib", compress=2)
>>>>>>> 05e89a8 (Added large file)
