# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from ucimlrepo import fetch_ucirepo 
  

def get_train_test_data():

    # fetch dataset 
    spambase = fetch_ucirepo(id=94) 
    
    # data (as pandas dataframes) 
    X = spambase.data.features 
    y = spambase.data.targets 
    
    # def naive_bayes_classifier():
    print("Creating trainning and testing data now ....")
    # Load the Iris dataset

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def naive_bayes_train_predict():
    print("Script is running now")
    X_train, X_test, y_train, y_test = get_train_test_data()
    # Initialize the Naive Bayes classifier (Gaussian Naive Bayes for this example)
    classifier = MultinomialNB()

    # Train the classifier on the training data
    classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = classifier.predict(X_test)

    # Evaluate the performance of the classifiers
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_str = classification_report(y_test, y_pred)

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report_str)

if __name__ == '__main__':
    # Run training task 
    naive_bayes_train_predict()
