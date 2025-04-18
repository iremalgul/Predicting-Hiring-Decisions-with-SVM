import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class CandidateSelector:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = SVC(kernel='linear')  # Default kernel is linear

    def generate_data(self, n=200, random_state=42):
        np.random.seed(random_state)
        experience = np.random.uniform(0, 10, n)
        score = np.random.uniform(0, 100, n)
        label = np.where((experience < 2) & (score < 60), 1, 0)

        self.df = pd.DataFrame({
            'experience_years': experience,
            'technical_score': score,
            'label': label
        })

    def split_and_scale(self):
        X = self.df[['experience_years', 'technical_score']]
        y = self.df['label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self, kernel_type='linear', C=1.0, gamma='scale'):
        """
        Train the SVC model with the specified kernel, C, and gamma.
        Default kernel is 'linear', but it can be changed to 'rbf', 'poly', or 'sigmoid'.
        """
        self.model = SVC(kernel=kernel_type, C=C, gamma=gamma)
        self.model.fit(self.X_train, self.y_train)
        print(f"Model trained with kernel='{kernel_type}', C={C}, gamma={gamma}")

    def tune_hyperparameters(self):
        """
        Tune hyperparameters C and gamma using GridSearchCV.
        """
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        }

        grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
        grid.fit(self.X_train, self.y_train)

        print("Best Parameters:", grid.best_params_)
        print("Best Score:", grid.best_score_)

        # Update the model with the best estimator
        self.model = grid.best_estimator_

    def evaluate_model(self):
        """
        Evaluate the trained model with accuracy score, confusion matrix, and classification report.
        """
        y_pred = self.model.predict(self.X_test)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))

    def plot_decision_boundary(self):
        """
        Plot the decision boundary of the model.
        """
        X = self.X_train
        y = self.y_train
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm')
        plt.xlabel("Experience (scaled)")
        plt.ylabel("Technical Score (scaled)")
        plt.title("SVC Decision Boundary")
        plt.savefig("decision_boundary.png", dpi=300)
        plt.close()

    def predict_candidate(self, experience_input, score_input):
        """
        Predict if a candidate will be hired or not based on experience and score.
        """
        input_df = pd.DataFrame([[experience_input, score_input]], columns=['experience_years', 'technical_score'])
        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)
        if prediction[0] == 1:
            print("The candidate will NOT be hired.")
        else:
            print("The candidate will be HIRED.")



