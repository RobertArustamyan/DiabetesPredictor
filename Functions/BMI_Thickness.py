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
    fill_bmi : bool
        If True, the class will fill missing BMI values; otherwise, it will fill missing SkinThickness values.

    Methods:
    -------
    fill():
        Trains a linear regression model and fills missing values in the DataFrame based on the model.
    """

    def __init__(self, dataframe: pd.DataFrame, fill_bmi: bool = True):
        self._df = dataframe
        self._fill_bmi = fill_bmi

    def _prepare_data(self) -> pd.DataFrame:
        """
        Prepares the data for training by dropping rows with NaN values in both 'BMI' and 'SkinThickness'.

        Returns:
        pd.DataFrame
            A DataFrame with rows containing non-null values for both columns.
        """
        train_data = self._df.dropna(subset=['BMI', 'SkinThickness'])
        return train_data

    @staticmethod
    def _train_model(X_train, y_train):
        """
        Trains a linear regression model.

        Parameters:
        X_train : pd.DataFrame
            The training features.
        y_train : pd.Series
            The target variable.

        Returns:
        LinearRegression
            The trained linear regression model.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def fill(self):
        """
        Trains a linear regression model and fills missing values in the DataFrame based on the model.
        """
        train_data = self._prepare_data()

        if self._fill_bmi:
            # Fill missing BMI values based on SkinThickness
            X_train = train_data[['SkinThickness']]
            y_train = train_data['BMI']
            model = Filler._train_model(X_train, y_train)

            missing_data = self._df[self._df['BMI'].isna()]
            X_missing = missing_data[['SkinThickness']]
            predictions = model.predict(X_missing)
            self._df.loc[self._df['BMI'].isna(), 'BMI'] = predictions
        else:
            # Fill missing SkinThickness values based on BMI
            X_train = train_data[['BMI']]
            y_train = train_data['SkinThickness']
            model = Filler._train_model(X_train, y_train)

            missing_data = self._df[self._df['SkinThickness'].isna()]
            X_missing = missing_data[['BMI']]
            predictions = model.predict(X_missing)
            self._df.loc[self._df['SkinThickness'].isna(), 'SkinThickness'] = predictions

    @property
    def get_dataframe(self):
        """
        Returns the processed DataFrame.

        Returns:
        pd.DataFrame
            The DataFrame with filled missing values.
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
        Removes rows where both 'BMI' and 'SkinThickness' are zero.
    _replace_zeroes():
        Replaces zero values with NaN in specified columns.
    fill():
        Calls _remove_zero_rows(), _replace_zeroes(), and then uses Filler to fill in missing values.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe

    def _remove_zero_rows(self):
        """
        Removes rows where both 'BMI' and 'SkinThickness' are zero.
        """
        self._df = self._df[~((self._df['BMI'] == 0) & (self._df['SkinThickness'] == 0))]

    def _replace_zeroes(self):
        """
        Replaces zero values with NaN in the specified columns of the DataFrame.
        """
        self._df.loc[self._df['BMI'] == 0, 'BMI'] = np.nan
        self._df.loc[self._df['SkinThickness'] == 0, 'SkinThickness'] = np.nan

    def fill(self):
        """
        Performs preprocessing and filling of missing values.
        """
        self._remove_zero_rows()
        self._replace_zeroes()

        # Fill missing BMI values
        bmi_filler = Filler(self._df, fill_bmi=True)
        bmi_filler.fill()

        # Fill missing SkinThickness values
        skin_filler = Filler(bmi_filler.get_dataframe, fill_bmi=False)
        skin_filler.fill()

        self._df = skin_filler.get_dataframe

    @property
    def get_dataframe(self):
        """
        Returns the processed DataFrame.

        Returns:
        pd.DataFrame
            The DataFrame with filled missing values.
        """
        return self._df


# Example usage
if __name__ == '__main__':
    try:
        df = pd.read_csv('../Data/diabetes.csv')
        filler_interface = FillerInterface(df)
        filler_interface.fill()
        filled_df = filler_interface.get_dataframe
        print(filled_df.head())
    except FileNotFoundError:
        print("The specified file was not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
