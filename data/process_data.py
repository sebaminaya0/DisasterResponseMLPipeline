import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Extracts data from the messages and categories files, then merge both to work with a unique DataFrame.

    Args:
        messages_filepath (str): file path of the messages file.
        categories_filepath (str): file path of the categories file.

    Returns:
        df (pandas.DataFrame): result DataFrame with the extracted information of messages and categories.
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories,left_on=['id'],right_on=['id'])

    return df


def clean_data(df):
    """Transforms DataFrame with all necessary steps.

    Args:
        df (pandas.DataFrame): DataFrame that needs cleaning.

    Returns:
        df (pandas.DataFrame): DataFrame already cleaned.
    """

    categories = df['categories'].str.split(';',expand=True)

    category_colnames = [x[0:-2] for x in list(categories.iloc[0,:])]

    categories.columns = category_colnames

    categories.replace('related-2','related-1',inplace=True)

    for column in categories:
        categories[column] = categories[column].astype(str).str.split("-").str.get(1)

        categories[column] = pd.to_numeric(categories[column])

    df.drop('categories', axis=1, inplace=True)

    df = pd.concat([df, categories], axis=1)

    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """Loads data into a SQLite database specified by the user.

    Args:
        df (pandas.DataFrame): DataFrame already cleaned.
        database_filename (str): filepath of the database.

    Returns:
        None
    """

    engine = create_engine('sqlite:///etl_done.db')
    df.to_sql('fact_messages', engine, index=False, if_exists='replace')

    return None


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
