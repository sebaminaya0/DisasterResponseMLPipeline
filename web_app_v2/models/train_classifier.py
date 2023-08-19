# import libraries

# libraries for data manipulation & loading
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle
import sys

# libraries for NLP
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

# libraries for ML
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """Loads all the data from the SQLite database.

    Args:
        database_filepath (str): name of the SQLite database.

    Returns:
        X: input for the ML model.
        Y: target for the ML model.
        categories: target column names.
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('fact_messages',engine)

    X = df['message'].values
    Y = df.iloc[:,4:].values
    categories = df.iloc[:,4:].columns

    return X, Y, categories


def tokenize(text):
    """Tokenizer for the tweet messages.

    Args:
        text (str): tweet message.

    Returns:
        tokens: list of lemmatized tokens, not including stopwords.
    """

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """ML model builder.

    Returns:
        cv: ML model instance.
    """

    pipeline_ = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__criterion': ["gini", "entropy"],
        'clf__estimator__n_jobs':[-1]
    }

    cv_ = GridSearchCV(pipeline_, parameters, n_jobs=-1)

    return cv_


def evaluate_model(model, X_test, Y_test, category_names):
    """ML model evaluator for all the categories of messages.

    Args:
        model: fitted ML model.
        X_test: test input data.
        Y_test: test target data.
        category_names: list with categories names.

    Returns:
        None
    """

    y_pred = model.predict(X_test)

    for i, element in enumerate(np.transpose(y_pred)):
        print("Category: " + category_names[i])
        print(" ")
        print(classification_report(np.transpose(y_pred)[i], np.transpose(Y_test)[i]))
        print(" ")


def save_model(model, model_filepath):
    """Saves the fitted ML model into a pickle file.

    Args:
        model: fitted ML model.
        model_filepath (str): filepath for the pickle file.

    Returns:
        None
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...\n\n')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
