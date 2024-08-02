import pandas as pd
import numpy as np

class GlucoseFiller:
    def __init__(self, df):
        self._df = df
        self._median_glucose_outcome_1 = df[(df['Outcome'] == 1) & (df['Glucose'] != 0)]['Glucose'].median()
        self._mode_glucose_outcome_0 = df[(df['Outcome'] == 0) & (df['Glucose'] != 0)]['Glucose'].mode()[0]
        self._fill_glucose()

    def _fill_glucose(self) -> None:
        """
        Fills glucose column with mode and median of Column nonzero values depending on outcome
        """
        for index,row in self._df.iterrows():
            if pd.isna(row['Glucose']) or row['Glucose'] == 0:
                if row['Outcome'] == 1:
                    self._df.at[index,'Glucose'] = self._median_glucose_outcome_1
                else:
                    self._df.at[index,'Glucose'] = self._mode_glucose_outcome_0
    @property
    def get_dataframe(self)-> pd.DataFrame:
        """
        Returns the Dataframe of Class
        :return: pd.Dataframe
        """

        return self._df

