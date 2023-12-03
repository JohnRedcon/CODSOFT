import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Loading the Titanic dataset
url = "/home/tshegofatso/Downloads/CodSoft Internship/Titanic/tested.csv"
titanic_data = pd.read_csv(url)

# Displaying the first few rows of the dataset
print(titanic_data.head())

# Checking if 'Survived' column is present
if 'Survived' not in titanic_data.columns:
    raise ValueError("Target variable 'Survived' not found in the dataset.")

# Separating features (X) and target variable (y)
X = titanic_data.drop(['Survived'], axis=1)
y = titanic_data['Survived']

# Preprocessing of the data
# Dropping irrelevant columns or columns with too many missing values
columns_to_drop = ['Name', 'Cabin', 'Ticket', 'PassengerId']
X = X.drop(columns_to_drop, axis=1)

# Separating numerical and categorical columns
numerical_cols = ['Age', 'Fare']
categorical_cols = ['Sex', 'Embarked']

# Imputing missing values for numerical columns
imputer_mean = SimpleImputer(strategy='mean')
X[numerical_cols] = imputer_mean.fit_transform(X[numerical_cols])

# Imputing missing values for categorical columns
imputer_mode = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = imputer_mode.fit_transform(X[categorical_cols])

# Converting categorical variables to numerical using label encoding
label_encoder = LabelEncoder()
X['Sex'] = label_encoder.fit_transform(X['Sex'])
X['Embarked'] = label_encoder.fit_transform(X['Embarked'])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building and training the RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Displaying the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
