import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load serving dataset
df = pd.read_csv("PM_POSHAN_Serving_Train_Modified.csv")

print("Dataset shape:", df.shape)
print(df.head())


# Features and target
features = [
    "Age",
    "Class",
    "Sex",
    "Height_cm",
    "Weight_kg",
    "Activity_Level",
    "Food_Selected"
]

target = "Serving_g"

X = df[features]
y = df[target]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Columns
numeric_features = [
    "Age",
    "Class",
    "Height_cm",
    "Weight_kg"
]

categorical_features = [
    "Sex",
    "Activity_Level",
    "Food_Selected"
]


# Preprocessing
preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_features),
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])


# Small Random Forest
model = RandomForestRegressor(
    n_estimators=40,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=1
)


# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# Train
print("\nTraining serving model...")
pipeline.fit(X_train, y_train)


# Evaluate
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel evaluation")
print("----------------")
print("MAE:", round(mae, 2), "g")
print("R2 :", round(r2, 4))


# Save model
joblib.dump(
    pipeline,
    "nutriai_serving_model.pkl",
    compress=3
)

print("\nModel saved as:")
print("nutriai_serving_model.pkl")