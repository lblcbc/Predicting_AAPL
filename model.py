# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("data_path")
data.iloc[63:73]

# %%
# Create new columns with historical values for each of the features
num_days = 62
features = ['mReturn', 'OpenClose', 'HighLow', 'Return', 'Volume']

for feature in features:
    for day in range(1, num_days+1):
        data[f'{feature}_{day}'] = data[feature].shift(day)

# Display the first few rows of the updated dataframe
data.head()



# %%
# Remove the first 63 rows
data = data.iloc[63:]

# Reset the index
data.reset_index(drop=True, inplace=True)

# Display the first few rows of the updated dataframe
data.head()


# %%
# Separate the features from the target
X = data.drop('Label', axis=1)
y = data['Label']

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
# Initialize the random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = classification_report(y_test, y_pred)
print(accuracy)
# 52% accuracy

# %%
# Select one of the trees in your random forest model
estimator = rf.estimators_[5]

plt.figure(figsize=(150,75))

# The plot_tree function readily plots a decision tree
tree.plot_tree(estimator, 
               feature_names=X_train.columns, 
               class_names=['mReturn', 'OpenClose', 'HighLow', 'Return', 'Volume'],
               filled=True)

plt.show()

# %%
# EVALUATING PERFORMANCE OF JUST PREDICTING IF TOMORROW MOVES UP OR DOWN
df = data
df['NextDayReturn'] = df['Return'].shift(-1)
df['Label'] = df['NextDayReturn'].apply(lambda x: 'UP' if x > 0 else 'DOWN')
df = df.iloc[:-1]
print(df.head())

# Define the feature set X and the target y
X = df.drop(['Label', 'NextDayReturn'], axis=1)
y = df['Label']
print(X)
print(y)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(X_test)

# Compute the performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

accuracy, precision, recall, f1
print(accuracy)
# 67% accruacy - still meh, considering trading fees and realistic frictions


# %%
# Select one of the trees in your random forest model
estimator = rf.estimators_[5]

plt.figure(figsize=(150,75))

# The plot_tree function readily plots a decision tree
tree.plot_tree(estimator, 
               feature_names=X_train.columns, 
               class_names=['UP', 'DOWN'],
               filled=True)

plt.show()

# %%
# Testing on Goldman Sachs, different sector to Apple
data = pd.read_csv("data_path")
df = data
num_days = 62
features = ['mReturn', 'OpenClose', 'HighLow', 'Return', 'Volume']

for feature in features:
    for day in range(1, num_days+1):
        df[f'{feature}_{day}'] = df[feature].shift(day)

df = df.iloc[63:]

df['NextDayReturn'] = df['Return'].shift(-1)

df['Labels'] = df['NextDayReturn'].apply(lambda x: 'UP' if x > 0 else 'DOWN')

df = df.iloc[:-1]

df.head()


# %%
X_gs = df.drop(['Labels', 'NextDayReturn'], axis=1)
y_gs = df['Labels']

y_pred = rf.predict(X_gs)

accuracy_gs = accuracy_score(y_gs, y_pred)
precision_gs = precision_score(y_gs, y_pred, average='macro')
recall_gs = recall_score(y_gs, y_pred, average='macro')
f1_gs = f1_score(y_gs, y_pred, average='macro')
print(accuracy_gs)
# 69%, decent


# %%
# Testing on ZM, which has fallen dramatrically since data start period (2020) to now (2023)
data = pd.read_csv("data_path")
df = data
num_days = 62
features = ['mReturn', 'OpenClose', 'HighLow', 'Return', 'Volume']

for feature in features:
    for day in range(1, num_days+1):
        df[f'{feature}_{day}'] = df[feature].shift(day)

df = df.iloc[63:]

df['NextDayReturn'] = df['Return'].shift(-1)

df['Labels'] = df['NextDayReturn'].apply(lambda x: 'UP' if x > 0 else 'DOWN')

df = df.iloc[:-1]

df.head()

# %%
X_zm = df.drop(['Labels', 'NextDayReturn'], axis=1)
y_zm = df['Labels']

y_pred = rf.predict(X_zm)

accuracy_zm = accuracy_score(y_zm, y_pred)
precision_zm = precision_score(y_zm, y_pred, average='macro')
recall_zm = recall_score(y_zm, y_pred, average='macro')
f1_gs = f1_score(y_zm, y_pred, average='macro')
print(accuracy_zm)
# 68%, decent


# %%
# Testing on LHA, European market
data = pd.read_csv("data_path")
df = data
num_days = 62
features = ['mReturn', 'OpenClose', 'HighLow', 'Return', 'Volume']

for feature in features:
    for day in range(1, num_days+1):
        df[f'{feature}_{day}'] = df[feature].shift(day)

df = df.iloc[63:]

df['NextDayReturn'] = df['Return'].shift(-1)

df['Labels'] = df['NextDayReturn'].apply(lambda x: 'UP' if x > 0 else 'DOWN')

df = df.iloc[:-1]

df.head()


X_lha = df.drop(['Labels', 'NextDayReturn'], axis=1)
y_lha = df['Labels']

y_pred = rf.predict(X_lha)

accuracy_lha = accuracy_score(y_lha, y_pred)
precision_lha = precision_score(y_lha, y_pred, average='macro')
recall_lha = recall_score(y_lha, y_pred, average='macro')
f1_lha = f1_score(y_lha, y_pred, average='macro')
print(accuracy_lha)
# 66%, decent


