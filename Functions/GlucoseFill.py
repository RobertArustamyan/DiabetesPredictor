import pandas as pd
import numpy as np
def fill_glucose(df: pd.DataFrame):
    # Calculate median and mode for the respective Outcome groups
    median_glucose_outcome_1 = df[df['Outcome'] == 1]['Glucose'].median()
    mode_glucose_outcome_0 = df[df['Outcome'] == 0]['Glucose'].mode()[0]
    # Apply the filling function with additional arguments
    df['Glucose'] = df.apply(
        fill,
        axis=1,
        median_glucose_outcome_1=median_glucose_outcome_1,
        mode_glucose_outcome_0=mode_glucose_outcome_0
    )

def fill(row, median_glucose_outcome_1, mode_glucose_outcome_0):
    if pd.isna(row['Glucose']):
        if row['Outcome'] == 1:
            return median_glucose_outcome_1
        elif row['Outcome'] == 0:
            return mode_glucose_outcome_0
    else:
        return row['Glucose']

if __name__ == '__main__':
    df = pd.read_csv('../Data/diabetes.csv')
    df.loc[df['Glucose'] == 0, 'Glucose'] = np.nan
    fill_glucose(df)
