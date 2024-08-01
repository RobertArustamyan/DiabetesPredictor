from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

class Filler:
    def __init__(self, dataframe: pd.DataFrame, fill_bmi: bool = True):
        self._df = dataframe
        self._fill_bmi = fill_bmi

    def _prepare_data(self):
        # Filter out rows with NaN values in both columns
        train_data = self._df.dropna(subset=['BMI', 'SkinThickness'])
        return train_data
    @staticmethod
    def _train_model(X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def fill(self):
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
        return self._df

class FillerInterface:
    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe

    def _remove_zero_rows(self):
        self._df = self._df[~((self._df['BMI'] == 0) & (self._df['SkinThickness'] == 0))]
    def _replace_zeroes(self):
        self._df.loc[self._df['BMI'] == 0, 'BMI'] = np.nan
        self._df.loc[self._df['SkinThickness'] == 0, 'SkinThickness'] = np.nan


    def fill(self):
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
        return self._df

if __name__ == '__main__':
    df = pd.read_csv('../Data/diabetes.csv')
    filler_interface = FillerInterface(df)
    filler_interface.fill()
    filled_df = filler_interface.get_dataframe
    print(filled_df.head())
