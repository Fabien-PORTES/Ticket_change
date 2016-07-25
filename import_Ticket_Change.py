# -*- coding: utf-8 -*-
"""
Éditeur de Spyder
"""
import pandas as pd
import os
import excelImport as imp

path = 'D:\\Users\\FPORTES\\Documents\\Ticket_ML\\CoC_Tickets_MachineLeaning\\'
files = [path + f for f in os.listdir(path) if "change" in f[0:6].lower()]
#file = path + "change_20151113_PM.xlsx"

#Les variables et leur format doivent etre renseigné dans les listes suivantes.

datetime_var = ["Request_For_Change_Time", "Scheduled_for_Approval_Time", "Scheduled_Start_Date",
               "Scheduled_End_Date", "Restart_After_Pending_Time", "Actual_End_Date",
               "INST_Task_Closed_Time"]
text_var = ["ID", "RQ_ID", "Type_of_Change", "EIST_Domain_ICT", "ApplicationCode",
            "ProgramName", "status", "Support_Group_Name+",
            "EIS_Ticket", "EIST_Domain_ICT", "EISTicket_CABDomain",
            "INST_Task_Name", "INST_Task_Assignee", "Requestor_Name",
            "Summary"]
numeric_var = ["Performance_Rating"]

variables = imp.columns()
variables.set_date_columns(datetime_var)
variables.set_text_columns(text_var)
variables.set_numeric_columns(numeric_var)

df_list = []

try:
    datafram = pd.read_csv(path + "merged.csv", sep = ";", index_col = 0)
except OSError:
    print("Le fichier merged n'existe pas.")
datafram['date_file'] = pd.to_datetime(datafram['date_file'],infer_datetime_format=True)
last_date =  datafram['date_file'].max()
print(last_date)

for f in files:
    print(f)
    date, AM_PM = imp.get_date_from_path(f)
    if date > last_date:
        df_list.append(imp.read_excel(f, variables))
    else:
        pass
#si les dataframes de df_list ot pas les memes colonnes, l'ordre des colonnes est cassé à la concaténation
df = pd.concat(df_list, axis = 0, ignore_index = True)
df = df.drop_duplicates(text_var + datetime_var + numeric_var)

df.to_csv(path_or_buf = path + "merged.csv", sep = ";", encoding = 'utf-8')

#print(df.describe(include = 'all'))

