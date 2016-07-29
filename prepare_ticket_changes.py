
# coding: utf-8
import sys
import pandas as pd
from unidecode import unidecode
import json
from matplotlib import interactive
interactive(True)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

to_parse = " ".join(sys.argv[1:])
user_input = json.loads(to_parse)
merged_data_path = user_input['merged_data_path']
save_path = user_input['save_path']
holiday_path = user_input['holiday_path']
equipe_path = user_input['equipe_path']

#merged_data_path = 'D:\\Users\\FPORTES\\Documents\\Ticket_ML\\CoC_Tickets_MachineLeaning\\merged.csv'
#save_path = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\cleaned_data_change_without_TM.csv"
#holiday_path = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\additional_data\\REFERENCE\\holidays.xlsx"
#equipe_path = "D:\\Users\\FPORTES\\Documents\\Ticket_ML\\additional_data\\REFERENCE\\Equipe.xlsx"

datetime_var = ["Request_For_Change_Time", "Scheduled_for_Approval_Time", "Scheduled_Start_Date",
               "Scheduled_End_Date", "Restart_After_Pending_Time", "Actual_End_Date",
               "INST_Task_Closed_Time", "date_file"]
text_var = ["Type_of_Change", "EIST_Domain_ICT", "ApplicationCode",
            "ProgramName", "status", "Support_Group_Name+",
            "EIS_Ticket", "EIST_Domain_ICT", "EISTicket_CABDomain",
            "INST_Task_Name", "INST_Task_Assignee", "Requestor_Name",
            "Summary", "ID", "RQ_ID", "AM_PM_file", "Performance_Rating"]
#variables catégorielles
categorical_var =[a for a in text_var if a not in ["ID", "RQ_ID", "Summary", "EIS_Ticket"]]
# liste des variables prédictives complété au fur et à mesure de leur preparation
predictors = []

#importation de fichier mergé des tickets changes
df_init = pd.read_csv(merged_data_path, sep = ";", index_col = 0)
df_init.shape

for var in text_var:
    df_init[var] = df_init[var].astype('category')
for var in datetime_var:
    df_init[var] = pd.to_datetime(df_init[var],infer_datetime_format=True)
#Des duplicats sur les ID, on garde la ligne ID oà la date Actual_end_date est la plus récente
df = df_init.sort_values(by =["ID", "Actual_End_Date"], ascending=False).groupby("ID").first()
df.shape
#les variables catégorielles sont décrétés catégorielles et les dates en timestamp


def rm_accents_tiret(data):
    """garde uniqueement les caractères ascii en minuscule et remplace les tirets par de espaces
    """
    data = str(data).replace("-", " ")
    data = str(data).replace("_", " ")
    return " ".join((unidecode(data)).lower().split())
df["INST_Task_Assignee_ascii"] = df["INST_Task_Assignee"].apply(rm_accents_tiret)

# importation des équipes et ajout de la variable ancienneté
df_equipe = pd.read_excel(equipe_path)
df_equipe["Année Début"].fillna(df_equipe["Année Début"].min(), inplace = True)
df_equipe["Année Fin"].fillna(df_equipe["Année Fin"].max(), inplace = True)
for index, row in df_equipe.iterrows():
    task_assignee = row["Prénom"] + " " + row["Nom"]
    task_assignee = rm_accents_tiret(task_assignee)
    df.loc[df["INST_Task_Assignee_ascii"] == task_assignee, "anciennete"] = row["Année Fin"] - row["Année Début"]

# importation des vacances et verification si elles ont lieu pendant la période prévisionelle de résolution d'un ticket

df_holiday = pd.read_excel(holiday_path)
col = ["Date debut","Date fin"]
for var in col:
    df_holiday[var] = pd.to_datetime(df_holiday[var],format ="%Y-%m-%d")

for index, row in df_holiday.iterrows():
    debut = row["Date debut"]
    fin = row["Date fin"]
    who = rm_accents_tiret(row["Who"])
    df["holiday"] = ((df["INST_Task_Assignee"] == who) & (#la date de début est entre le début et la fin des dates prévisionelles
            ((df['Scheduled_Start_Date']<= debut) & (df['Scheduled_End_Date'] >= debut)) | \
            #la date de fin est entre le début et la fin des dates prévisionelles
            ((df['Scheduled_Start_Date']<= fin) & (df['Scheduled_End_Date'] >= fin)))) * 1
#df["holiday"].describe()

from workingTime import working_time, working_day
#construction variable cible : 1 si ticket défaillant, 0 sinon !

# Défaillant si délai de traitement supérieur à 14h sans compter les week end et uniquement pendant les
# heures de travail (8h-18h)
SLA = pd.Timedelta(hours = 14)
df["delay_14h"] = df.apply(working_time, axis = 1)
df['delay_14h_bin'] = 1 * (df["delay_14h"] > SLA)
# Défaillant si délai de traitement supérieur à 48h sans compter les week end
SLA = pd.Timedelta(days = 2)
df["delay_48h"] = df.apply(working_day, axis = 1)
df['delay_48h_bin'] = 1 * (df["delay_48h"] > SLA)

#print("yolo", working_time(d))
#cible en tenant compte des heures de travail. ne donne rien
df["delay"] = df.apply(working_time, axis = 1)
df['delay_bin'] = 1 * (df["delay"] > SLA)

#La date de soumission du ticket est éclaté
df["weekday_Request_For_Change_Time"] = df["Request_For_Change_Time"].apply(lambda x: x.weekday())
df["month_Request_For_Change_Time"] = df["Request_For_Change_Time"].apply(lambda x: x.month)
df["year_Request_For_Change_Time"] = df["Request_For_Change_Time"].apply(lambda x: x.year)
df["hour_Request_For_Change_Time"] = df["Request_For_Change_Time"].apply(lambda x: x.hour)
predictors.extend(["weekday_Request_For_Change_Time", "month_Request_For_Change_Time",
                  "year_Request_For_Change_Time", "hour_Request_For_Change_Time"])
#extract informations from labels avec des expressions regulières
df[["TOC_code","TOC_level"]] = df['Type_of_Change'].str.extract("(?:.*?)-(?P<TOC_code>.*?)\s*(?:-\s*(?:.*?))*(?:[/|-](?P<TOC_level>.*))", expand = True)
df["TOC_level"] = df["TOC_level"].str.strip()
#on garde uniquement les valeurs high low et medium
df["TOC_level"] = df[df["TOC_level"].isin(['M', 'L', 'H', 'Medium', 'Low', 'High'])]['TOC_level']
df["TOC_level"] = df["TOC_level"].str[0]

df[["INST_techno_gpe","INST_desc_techno"]] = df['INST_Task_Name'].str.extract("(?:\w)_(?P<INST_techno_gpe>.*?)\s*_\s*(?P<INST_desc_techno>.*)", expand = True)

#df['EIST_Domain_ICT_reduced'] = df['EIST_Domain_ICT'].str[0:2]

categorical_var += ["TOC_code","TOC_level","INST_techno_gpe","INST_desc_techno"]


for var in categorical_var:
    df[var] = df[var].astype('category')

def plot(df, var):
    """ plot en barre en comptant les missings
    """
    m = df[var].value_counts(normalize=False, dropna=False).sort_values(ascending = False)
    f = m.plot(kind = 'bar')

def recode_inf_5(dfme, var):
    """crée une nouvelle variable dans la dataframe df qui represente la variable var
    où les modalités dont l'effectif est inferieur à 5% ont été regroupe sous le label Inf_5_percent
    les valeurs manquantes sont remplacé par la valeur "Missing" et ne sont pas regroupé meme si inf a 5%
    """
    var_recoded = var + "_Modified"
    mod_inf_5 = dfme[var].value_counts()[dfme[var].value_counts(normalize=True, dropna=False) < 0.05].index.tolist()
    dfme[var_recoded] = dfme[var].astype("object")
    dfme[var_recoded].fillna("Missing",  inplace=True)
    dfme.loc[dfme[var_recoded].isin(mod_inf_5), var_recoded] = "Inf_5"
    dfme[var_recoded] = dfme[var_recoded].astype('category')
    return dfme[var_recoded]
def missing(dfme, var):
    """replace missing par le label missing"""
    var_recoded = var + "_Modified"
    dfme[var_recoded] = dfme[var].astype("object")
    dfme[var_recoded].fillna("Missing",  inplace=True)
    return dfme[var_recoded]


#Performance_Rating
var = "Performance_Rating"
var_recoded = var + "_Modified"
#plot(df, var)
df[var_recoded] = pd.to_numeric(df[var])
df[var_recoded] = ((df[var_recoded]< 3) & (df[var_recoded] >= 0)) * 1 + (df[var_recoded]>= 3) * 2
df[var_recoded] = df[var_recoded].astype('category')
df[var_recoded].cat.categories = ["Missing", "Unsatisfied", "Satisfied"]
#plot(df, var_recoded)
predictors.append(var_recoded)

#INST_techno_gpe
var = "INST_techno_gpe"
var_5 = var + "_Modified"
df[var_5] = recode_inf_5(df, var)
#plot(df, var_5)
predictors.append(var_5)

#INST_desc_techno
var = "INST_desc_techno"
var_5 = var + "_Modified"
df[var_5] = missing(df, var)
#plot(df, var_5)
predictors.append(var_5)

#TOC_code
var = "TOC_code"
var_5 = var + "_Modified"
#df[var_5] = recode_inf_5(df, var)
df[var_5] = df[var].str.extract("(?P<TOC_code_reduced>.+)\d", expand = True)
df[var_5].fillna("Missing", inplace = True)
df[var_5] = df[var_5].astype('category')
#plot(df, var_5)
predictors.append(var_5)

#TOC_level
var = "TOC_level"
var_5 = var + "_Modified"
df[var_5] = recode_inf_5(df, var)
#plot(df, var_5)
predictors.append(var_5)

#EIST_Domain_ICT_reduced
var = "EIST_Domain_ICT"
var_recoded = var + "_Modified"
df[var_recoded] = missing(df, var)

#mod_inf_5 = df[var].value_counts()[df[var].value_counts(normalize=True, dropna=False) < 0.05].index.tolist()
#df[var_recoded] = df[var].astype("object")
#df[var_recoded].fillna("Missing",  inplace=True)
#df.loc[df[var_recoded].isin(l for l in mod_inf_5 if l.lower() != "s"), var_recoded] = "Inf_5"
#df[var_recoded] = df[var_recoded].astype('category')
#plot(df, var_recoded)
predictors.append(var_recoded)

#INST_Task_Assignee
var = "INST_Task_Assignee"
var_5 = var + "_Modified"
df[var_5] = missing(df, var)
#plot(df, var_5)
predictors.append(var_5)

#Support_Group_Name+
var = "Support_Group_Name+"
var_5 = var + "_Modified"
df[var_5] = recode_inf_5(df, var)
#plot(df, var_5)
predictors.append(var_5)

#anciennete
var = "anciennete"
var_recoded = var + "_Modified"
df[var_recoded] = ((df[var]< 5) & (df[var] >= 0)) * 1 + (df[var] == 5) * 2
df[var_recoded] = df[var_recoded].astype('category')
df[var_recoded].cat.categories = ["Missing", "< 5ans", "5 ans"]
#plot(df, var_recoded)
predictors.append(var_recoded)

#INST_Task_Assignee
var = "INST_Task_Assignee"
var_recoded = var + "_Modified"
df[var_recoded] = missing(df, var)
#plot(df, var_recoded)
predictors.append(var_recoded)

#Liste de predicteurs
predictors += ["Request_For_Change_Time", "Summary"]
to_save = predictors + ["delay_48h_bin", "delay_14h_bin"]

df["Summary"] = df["Summary"].apply(rm_accents_tiret)

to_save = list(set(to_save))
print(to_save)
df[to_save].to_csv(save_path, sep = ";", encoding = 'utf-8')




