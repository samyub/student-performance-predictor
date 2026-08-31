import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_model():

    data = pd.read_csv("data/student_data.csv")

    features = [
        "study_hours",
        "attendance",
        "assignments_completed",
        "previous_score"
    ]

    X = data[features]
    y = data["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy
