# -*- coding: utf-8 -*-
"""
working_time calcule le temps de travail entre 2 dates.
Les 2 dates sont dans les colonnes ayant les noms de fromDate et toDate
working_time est une fonction a passer à la méthode apply d'une dataframe
df.apply(working_time, axis = 1)

@author: fportes
"""
import pandas as pd
import datetime
import numpy as np

# les deux colonnes de la dataframe entre lesquelles doit etre calculé le temps de travail
fromDate = "Request_For_Change_Time"
toDate = "Scheduled_for_Approval_Time"

def working_from_hours(date):
    if date.hour >= 18:
        delta = pd.Timedelta(0)
    elif date.hour < 8:
        delta = pd.Timedelta(hours = 10)
    else:
        delta = datetime.datetime(date.year, date.month, date.day, 18, 0, 0) - date
    return delta        
        
def working_to_hours(date):
    if date.hour < 8:
        delta = pd.Timedelta(0)
    elif date.hour >= 18:
        delta = pd.Timedelta(hours = 10)
    else :
        delta = date - datetime.datetime(date.year, date.month, date.day, 8, 0, 0)
    return delta

def working_time(x):
    delta_day = pd.Timedelta(0)
    fromdate = x["Request_For_Change_Time"]
    todate = x["Scheduled_for_Approval_Time"]
    if str(fromdate) == "NaT" or str(todate) == "NaT":
        return np.nan
    delta1 = working_from_hours(fromdate)
    delta2 = working_to_hours(todate)
    daygenerator = (fromdate + datetime.timedelta(x + 1) for x in range((todate - fromdate).days))
    working_day_hour = (sum(1 for day in daygenerator if day.weekday() < 5) -1) * 10
    delta_day = pd.Timedelta(0)
    if working_day_hour > 0:
        delta_day = pd.Timedelta(hours = working_day_hour) 
    return delta1 + delta_day + delta2

def working_day(x):
    fromdate = x["Request_For_Change_Time"]
    todate = x["Scheduled_for_Approval_Time"]
    if str(fromdate) == "NaT" or str(todate) == "NaT":
        return np.nan
    daygenerator = (fromdate + datetime.timedelta(x + 1) for x in range((todate - fromdate).days))
    non_working_days = sum(1 for day in daygenerator if day.weekday() > 4)
    delta = todate - fromdate - pd.Timedelta(days = non_working_days )
    return delta