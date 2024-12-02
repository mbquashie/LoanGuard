import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

def train_random_forest(X_train, y_train, X_test, y_test):
    print("Training Random Forest...")
    rfc = RandomForestClassifier()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    scores = cross_val_score(rfc, X_train, y_train, scoring='accuracy', cv=folds)
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}, Std: {scores.std():.4f}")
    
    start = time.time()
    rfc.fit(X_train, y_train)
    stop = time.time()
    print(f"Training time: {stop - start:.2f}s")

    predictions = rfc.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, predictions):.4f}")

    # Save the model
    joblib.dump(rfc, 'random_forest_model.h5')
    print("Model saved as 'random_forest_model.h5'")

