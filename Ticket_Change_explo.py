
# coding: utf-8

# In[107]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import xlrd, os, datetime
import collections
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


# In[108]:

path = 'D:\\Users\\FPORTES\\Documents\\Ticket_ML\\CoC_Tickets_MachineLeaning\\'
files = [path + f for f in os.listdir(path) if "change" in f[0:6].lower()]
file = path + "merged.csv"

datetime_var = ["Request_For_Change_Time", "Scheduled_for_Approval_Time", "Scheduled_Start_Date",
               "Scheduled_End_Date", "Restart_After_Pending_Time", "Actual_End_Date",
               "INST_Task_Closed_Time", "date_file"]
text_var = ["Type_of_Change", "EIST_Domain_ICT", "ApplicationCode",
            "ProgramName", "status", "Support_Group_Name+",
            "EIS_Ticket", "EIST_Domain_ICT", "EISTicket_CABDomain",
            "INST_Task_Name", "INST_Task_Assignee", "Requestor_Name",
            "Summary", "ID", "RQ_ID", "AM_PM_file"]
numeric_var = ["Performance_Rating"]


# In[109]:

writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
df = pd.read_csv(file, sep = ";", index_col = 0)
df.shape


# In[110]:

for var in text_var:
    df[var] = df[var].astype('category')
for var in datetime_var:
    df[var] = pd.to_datetime(df[var],infer_datetime_format=True)
df.dtypes


# In[111]:

df.describe(include = 'all').to_clipboard()
df['Support_Group_Name+'].value_counts().plot(kind = 'pie')


# In[87]:

df.groupby("ID").first()['status'].value_counts().plot(kind = 'bar')
for var in text_var:
    if "ID" in var:
        pass
    else:
        a = df.groupby("ID").first()[var].value_counts().plot(kind = 'bar')
        a.figure.savefig("D:\\Users\\FPORTES\\Documents\\Ticket_ML\\plot\\" + var + ".pdf")


# In[112]:

date = df[df.duplicated("ID", keep = False)].sort_values("ID")


# In[113]:

date_min = date[["ID"] + datetime_var].groupby("ID").min()
date_max = date[["ID"] + datetime_var].groupby("ID").max()
d = date_max - date_min
d[datetime_var].max()


# In[114]:

d.sort_values(datetime_var, ascending=False).to_clipboard()


# In[115]:

d.describe()


# In[117]:

df['target'] = df["Scheduled_End_Date"] - df["Actual_End_Date"]


# In[129]:

df["target"].describe()
df[df['target'].notnull()]['target']


# In[98]:

df['EIS_Ticket'].mode().iloc[0]


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



