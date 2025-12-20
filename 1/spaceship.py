import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# -----------------------------
# Feature Engineering Function
# -----------------------------
def feature_engineering(df):
    # Split PassengerId into group and member number
    df[['Group', 'GroupMember']] = df['PassengerId'].str.split('_', expand=True)
    df['GroupMember'] = df['GroupMember'].astype(int)

    # Split Cabin into Deck / Number / Side
    if 'Cabin' in df.columns:
        df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
        df.drop(columns=['Cabin'], inplace=True)

    # Create TotalSpent from various spending categories
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in spend_cols:
        if col not in df.columns:
            df[col] = 0
    df['TotalSpent'] = df[spend_cols].sum(axis=1)

    # Group size and IsAlone feature
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('count')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

    # Age binning - convert to string immediately to avoid dtype issues later
    df['AgeBin'] = pd.cut(df['Age'], bins=[-1, 12, 18, 35, 60, 100], 
                          labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']).astype(str)

    return df

# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Save PassengerId for submission
test_passenger_ids = test_df['PassengerId']

# Apply feature engineering
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Add 'source' to distinguish between train and test
train_df['source'] = 'train'
test_df['source'] = 'test'
test_df['Transported'] = pd.NA  # Placeholder for target in test set

# Combine train and test
full_df = pd.concat([train_df, test_df], ignore_index=True)

# -----------------------------
# Handle categorical and numeric columns
# -----------------------------
# Save 'source' before encoding
source_col = full_df['source']

# Identify columns
cat_cols = full_df.select_dtypes(include='object').columns.tolist()
cat_cols = [col for col in cat_cols if col != 'source']
num_cols = full_df.select_dtypes(include='number').columns.tolist()

# Impute numeric columns
num_imputer = SimpleImputer(strategy='median')
full_df[num_cols] = num_imputer.fit_transform(full_df[num_cols])

# Impute categorical columns
full_df[cat_cols] = full_df[cat_cols].replace({pd.NA: np.nan})
cat_imputer = SimpleImputer(strategy='most_frequent')
full_df[cat_cols] = cat_imputer.fit_transform(full_df[cat_cols])

# Convert categorical columns to string type explicitly after imputation
for col in cat_cols:
    full_df[col] = full_df[col].astype(str)

# Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    full_df[col] = le.fit_transform(full_df[col])
    label_encoders[col] = le

# Restore 'source' column (as string)
full_df['source'] = source_col

# -----------------------------
# Split back into train/test
# -----------------------------
train_df = full_df[full_df['source'] == 'train'].copy()
test_df = full_df[full_df['source'] == 'test'].copy()

# Convert target to boolean
train_df['Transported'] = train_df['Transported'].astype(bool)

# Drop unused columns
train_df.drop(columns=['source'], inplace=True)
test_df.drop(columns=['source', 'Transported'], inplace=True)

# -----------------------------
# Prepare for modeling
# -----------------------------
X = train_df.drop(columns=['Transported', 'PassengerId'])
y = train_df['Transported']

# Sanity check
if X.shape[0] == 0:
    raise ValueError("No training samples found!")

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Decision Tree Model
# -----------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds_val = dt_model.predict(X_val)
dt_accuracy = accuracy_score(y_val, dt_preds_val)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

# Predict on test set
dt_test_preds = dt_model.predict(test_df.drop(columns=['PassengerId']))
dt_submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Transported': dt_test_preds.astype(bool)
})
dt_submission.to_csv('submission_decision_tree.csv', index=False)

# -----------------------------
# Random Forest Model
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds_val = rf_model.predict(X_val)
rf_accuracy = accuracy_score(y_val, rf_preds_val)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Predict on test set
rf_test_preds = rf_model.predict(test_df.drop(columns=['PassengerId']))
rf_submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Transported': rf_test_preds.astype(bool)
})
rf_submission.to_csv('submission_random_forest.csv', index=False)