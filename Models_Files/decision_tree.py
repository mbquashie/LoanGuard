import joblib
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

def train_decision_tree(X_train, y_train, X_test, y_test):
    print("Training Decision Tree...")
    dtc = DecisionTreeClassifier()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    scores = cross_val_score(dtc, X_train, y_train, scoring='accuracy', cv=folds)
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}, Std: {scores.std():.4f}")
    
    start = time.time()
    dtc.fit(X_train, y_train)
    stop = time.time()
    print(f"Training time: {stop - start:.2f}s")

    predictions = dtc.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, predictions):.4f}")

    # Save the model
    joblib.dump(dtc, 'decision_tree_model.h5')
    print("Model saved as 'decision_tree_model.h5'")

