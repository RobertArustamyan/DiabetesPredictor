from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


class Filler:
    """
    A class to handle the filling of missing values in a DataFrame using linear regression.

    Attributes:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing data with potential missing values.
    columns_pair : list of lists
        A list of column pairs to be processed for filling missing values.

    Methods:
    -------
    fill():
        Trains linear regression models and fills missing values in the DataFrame based on the models.
    """

    def __init__(self, dataframe: pd.DataFrame, columns_pair: list):
        self._df = dataframe
        self._columns_pair = columns_pair

    def _prepare_data(self, columns):
        """
        Prepares the data for training by dropping rows with NaN values in the specified columns.

        Parameters:
        columns : list
            The columns to check for non-null values.

        Returns:
        pd.DataFrame
            A DataFrame with rows containing non-null values for the specified columns.
        """
        return self._df.dropna(subset=columns)

    @staticmethod
    def _train_model(X_train:pd.DataFrame, y_train:pd.DataFrame):
        """
        Trains a linear regression model.

        :param X_train: The training features.
        :param y_train: The target variable.
        :return: The trained linear regression model.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def fill(self):
        """
        Trains linear regression models and fills missing values in the DataFrame based on the models.
        """
        for columns in self._columns_pair:
            train_data = self._prepare_data(columns)

            if len(columns) == 2:
                feature_col, target_col = columns

                # Check which column to fill
                if self._df[feature_col].isna().any():
                    # Fill missing values for feature_col based on target_col
                    X_train = train_data[[target_col]]
                    y_train = train_data[feature_col]
                    model = Filler._train_model(X_train, y_train)

                    missing_data = self._df[self._df[feature_col].isna()]
                    X_missing = missing_data[[target_col]]
                    predictions = model.predict(X_missing)
                    self._df.loc[self._df[feature_col].isna(), feature_col] = predictions

                if self._df[target_col].isna().any():
                    # Fill missing values for target_col based on feature_col
                    X_train = train_data[[feature_col]]
                    y_train = train_data[target_col]
                    model = Filler._train_model(X_train, y_train)

                    missing_data = self._df[self._df[target_col].isna()]
                    X_missing = missing_data[[feature_col]]
                    predictions = model.predict(X_missing)
                    self._df.loc[self._df[target_col].isna(), target_col] = predictions

    @property
    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the Dataframe of Class
        :return: pd.Dataframe
        """
        return self._df


class FillerInterface:
    """
    A class to interface with the Filler class, performing preprocessing and filling operations.

    Attributes:
    ----------
    dataframe : pd.DataFrame
        The DataFrame to be processed.

    Methods:
    -------
    _remove_zero_rows():
        Removes rows where both columns are zero.
    _replace_zeroes():
        Replaces zero values with NaN in specified columns.
    fill():
        Calls _remove_zero_rows(), _replace_zeroes(), and then uses Filler to fill in missing values.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe

    def _remove_zero_rows(self) -> None:
        """
        Removes rows where both 'BMI' and 'SkinThickness' or other column pairs are zero.
        """
        self._df = self._df[~((self._df[['BMI', 'SkinThickness']].eq(0).all(axis=1)) |
                              (self._df[['Glucose', 'Insulin']].eq(0).all(axis=1)))]

    def _replace_zeroes(self) -> None:
        """
        Replaces zero values with NaN in specified columns of the DataFrame.
        """
        self._df.loc[self._df['BMI'] == 0, 'BMI'] = np.nan
        self._df.loc[self._df['SkinThickness'] == 0, 'SkinThickness'] = np.nan
        self._df.loc[self._df['Glucose'] == 0, 'Glucose'] = np.nan
        self._df.loc[self._df['Insulin'] == 0, 'Insulin'] = np.nan

    def fill(self) -> None:
        """
        Fills the missing values for BMI, SkinThickness, Glucose, Insulin columns
        """
        self._remove_zero_rows()
        self._replace_zeroes()

        # Define column pairs to be processed
        columns_pairs = [['BMI', 'SkinThickness'], ['Glucose', 'Insulin']]

        # Fill missing values for each column pair
        filler = Filler(self._df, columns_pair=columns_pairs)
        filler.fill()

        # Update the DataFrame with filled values
        self._df = filler.get_dataframe

    @property
    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the Dataframe of Class
        :return: pd.Dataframe
        """
        return self._df


# Example usage
if __name__ == '__main__':
    try:
        # Load the DataFrame from a CSV file
        df = pd.read_csv('../Data/diabetes.csv')
        # Initialize FillerInterface with the DataFrame
        filler_interface = FillerInterface(df)
        # Perform the filling process
        filler_interface.fill()
        # Get the filled DataFrame
        filled_df = filler_interface.get_dataframe
        print(filled_df.head())
    except FileNotFoundError:
        print("The specified file was not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
