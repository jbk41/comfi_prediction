import numpy as np
import json
from pandas import DataFrame
from sklearn import linear_model


def json_to_data(data_string):
    parsed_json = json.loads(data_string)

    spend_save = {}
    spend_save['scaled_monthly_spending_N'] = []
    spend_save['scaled_monthly_saving_N']   = []
    spend_save['scaled_monthly_spending_Nm1'] = []
    spend_save['scaled_monthly_saving_Nm1']   = []


    for entry in parsed_json['data']:
        spend_save['scaled_monthly_spending_N'].append(entry['scaled_monthly_spending'])
        spend_save['scaled_monthly_saving_N'].append(entry['scaled_monthly_saving'])
        spend_save['scaled_monthly_spending_Nm1'].append(entry['scaled_monthly_spending'])
        spend_save['scaled_monthly_saving_Nm1'].append(entry['scaled_monthly_saving'])

    del spend_save['scaled_monthly_spending_Nm1'][-1]
    del spend_save['scaled_monthly_saving_Nm1'][-1]
    spend_save['scaled_monthly_spending_N'].pop(0)
    spend_save['scaled_monthly_saving_N'].pop(0)

    average_monthly_income = parsed_json['average_monthly_income']

    return spend_save



if __name__ == '__main__':

    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description='Spend and save predictor')
    parser.add_argument("data_string",
                        help="'expecting json data from sim'")
    args = parser.parse_args()

    spend_save = json_to_data(args.data_string)
    df = DataFrame(spend_save,columns=['scaled_monthly_spending_N','scaled_monthly_saving_N','scaled_monthly_spending_Nm1','scaled_monthly_saving_Nm1']) 
    X = df[['scaled_monthly_spending_Nm1','scaled_monthly_saving_Nm1']] # independent variables
    save = df['scaled_monthly_saving_N'] # predict savings
    spend = df['scaled_monthly_spending_N'] # predict spending

    average_spend = np.mean(spend_save['scaled_monthly_spending_N'])
    average_save = np.mean(spend_save['scaled_monthly_saving_N'])

    save_regr = linear_model.LinearRegression()
    save_regr.fit(X, save)

    spend_regr = linear_model.LinearRegression()
    spend_regr.fit(X, spend)
    print('this_month_saving = intercept + a * previous_month_spending + b * previous_month_saving')
    print('Saving Prediction:')
    print('Intercept: \n', save_regr.intercept_)
    print('Coefficients: \n', save_regr.coef_)
    
    print('\nthis_month_spending = intercept + a * previous_month_spending + b * previous_month_saving')
    print('Spending Prediction:')
    print('Intercept: \n', spend_regr.intercept_)
    print('Coefficients: \n', spend_regr.coef_)
    #    #print('this_month_spending = ' + spend_regr.intercept_ + ' + ' + spend_regr.coef_[0]  + '*previous_month_spending + ' + spend_regr.coef_[1] + '*previous_month_saving')









