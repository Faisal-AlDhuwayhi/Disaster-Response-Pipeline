# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge the messages and categories data

    Args:
        messages_filepath ([string]): file path of the messages data file
        categories_filepath ([string]): file path of the categories data file

    Returns:
        dataframe: a merged dataframe of the messages and categories data
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = categories.merge(messages, on=['id'])

    return df


def clean_data(df):
    """Clean and reorganize the data 

    Args:
        df ([dataframe]): a dataframe containing the data

    Returns:
        dataframe: the dataframe after cleaning
    """
    # split categories column into separate category columns
    categories = df['categories'].str.split(';', expand=True)

    # extract the columns names from the first row
    row = categories.iloc[0, :]
    category_colnames = list(row.apply(lambda x: x[:-2]))

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string (which is the number)
        categories[column] = categories[column].astype(str).str[-1]
        
        categories[column] = pd.to_numeric(categories[column])

    # the related category has small amount of value 2. It is converted to the majority class which is one,
    # to have only 1 and 0 values.
    categories['related'] = categories['related'].apply(lambda x: 1 if x == 2 else x)

    # replace categories column in df with new category columns.
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # remove duplicates
    df = df[~df.duplicated()]

    return df


def save_data(df, database_filename):
    """Save the dataframe to a database

    Args:
        df ([dataframe]): a dataframe containing the data
        database_filename ([string]): a database file path
    """
    
    # create connection with the database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # upload the dataframe to the database
    df.to_sql('disaster_response_table', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()