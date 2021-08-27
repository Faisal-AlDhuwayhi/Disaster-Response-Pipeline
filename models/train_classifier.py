# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import sys
import re
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """Load the data from the database and select features and target

    Args:
        database_filepath ([string]): a database file path

    Returns:
        [dataframe, array]: a dataframe that contains the features data
        [dataframe, array]: a dataframe that contains the target data
        [list]: a list that contains the category names
    """
    # create connection with the database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    # upload the table to a dataframe
    df = pd.read_sql_table('disaster_response_table', engine)

    # select features and target
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    # get category names
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """Tokenize and clean the text data 

    Args:
        text ([string]): a text message   

    Returns:
        [list]: a list of cleaned tokens of the text
    """
    # normalize case and strip additional spaces
    text = text.lower().strip()
    
    # remove punctuations
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    clean_tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
        
    return clean_tokens


def build_model():
    """Build the model with pipeline and grid search

    Returns:
        [object]: a grid search object that contains the model
    """
    # create ML pipline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # testing hyperparameters
    parameters = {
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [2, 5]
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model with the common classification metrics

    Args:
        model ([object]): a multi-output classification model
        X_test ([array]): the features test data
        Y_test ([array]): the target test data
        category_names ([list]): a list of category names of the target
    """
    # predict the test set
    y_preds = model.predict(X_test)

    # print classification report for each category
    print(classification_report(Y_test, y_preds, target_names=category_names))

    # print accuracy metric
    accuracy = (y_preds == Y_test.values).mean()
    print('\nThe model accuracy: {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    """Save the model into a specific pickle file 

    Args:
        model ([object]): a multi-output classification model
        model_filepath ([string]): the model file path
    """
    # save the model into a pickle file
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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
        
        print('Evaluating model...')
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