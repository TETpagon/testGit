import pandas as pd
from pprint import pprint as pp
from config import config
from datetime import datetime


def readXLSX(path):
    return pd.read_excel(path, sheet_name=None)


def getDebit():
    df = readXLSX(config.pathToDebit_15795)
    februaly_well_15795 = df['threadRpSheet'][['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 3', 'Unnamed: 4']].loc[26:53]
    februaly_well_15795.rename(
        columns={'Unnamed: 0': 'date', 'Unnamed: 1': 'time', 'Unnamed: 3': 'time_interval', 'Unnamed: 4': 'debit'},
        inplace=True)
    februaly_well_15795 = februaly_well_15795.reset_index()[['date', 'time', 'time_interval', 'debit']]

    df = readXLSX(config.pathToDebit_18073)
    februaly_well_18073 = df['threadRpSheet'][['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 3', 'Unnamed: 4']].loc[36:62]
    februaly_well_18073.rename(
        columns={'Unnamed: 0': 'date', 'Unnamed: 1': 'time', 'Unnamed: 3': 'time_interval', 'Unnamed: 4': 'debit'},
        inplace=True)
    februaly_well_18073 = februaly_well_18073.reset_index()[['date', 'time', 'time_interval', 'debit']]

    df = readXLSX(config.pathToDebit_30065)
    februaly_well_30065 = df['threadRpSheet'][['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 3', 'Unnamed: 4']].loc[8:70]
    februaly_well_30065.rename(
        columns={'Unnamed: 0': 'date', 'Unnamed: 1': 'time', 'Unnamed: 3': 'time_interval', 'Unnamed: 4': 'debit'},
        inplace=True)
    februaly_well_30065 = februaly_well_30065.reset_index()[['date', 'time', 'time_interval', 'debit']]
    # pd.DataFrame.sort_values(ascending=)

    return {'well_15795': februaly_well_15795, 'well_18073': februaly_well_18073, 'well_30065': februaly_well_30065}


if __name__ == "__main__":
    listDebit = getDebit()
