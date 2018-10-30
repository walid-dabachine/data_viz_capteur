import pandas as pd
import numpy as np
import datetime
import re
import matplotlib.pyplot as plt
import glob
import os
from scipy import stats
from dateutil import rrule, parser
import matplotlib.dates as mdates
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode, plot
from plotly import tools
init_notebook_mode(connected=False)
pd.set_option('display.max_rows', False)
pd.set_option('display.max_columns', 100)

def stats_desc_df(df, filepath, file_missing_values):
    print(filepath, file=file_missing_values)
    check_na_values(df, file_missing_values)
    print(df.describe(), file=file_missing_values)
    print("\n\n", file=file_missing_values)

def load_and_clean_nancy(file_path, xlsx_sheet_number, time_column_name, file_missing_values):
    
    #Chargement du fichier
    df_nancy = pd.read_excel(io=file_path, sheet_name=xlsx_sheet_number, header=1)
    #Suppression de la 1ere ligne inutile
    df_nancy = df_nancy[1:]
    #Definition de la colonne du timestamp
    df_nancy = df_nancy.rename(columns={df_nancy.columns[0]: time_column_name })
    #Definition du type des colonnes (sous forme de dict) + formatage de la colonne timestamp
    columns = list(df_nancy.columns)
    types_list = [datetime.datetime, np.float32, np.float32, np.float32]
    types_dict = dict(zip(columns,types_list))
    df_nancy = df_nancy.astype(types_dict)
    df_nancy[time_column_name]=pd.to_datetime(df_nancy[time_column_name], format="%Y-%m-%d %H:%M")
    df_nancy = df_nancy.set_index(time_column_name)
    stats_desc_df(df_nancy, file_path, file_missing_values)
    
    return df_nancy

def check_na_values(df, file_missing_values):
    bool_na_values = False
    for col in df:
        sum_nbr = df[col].isna().sum()
        if sum_nbr != 0:
            bool_na_values = True
        perc = round((sum_nbr/len(df))*100, 2)
        print(col, ":", str(sum_nbr),"(",perc,"%)", file=file_missing_values)
    print("\n", file=file_missing_values)
    return bool_na_values

def interpolate_dataframe(df, method, axis, limit, limit_area, limit_direction="forward"):
    if (len(df._get_numeric_data().columns)) == len(df.columns):
        print("All columns are numeric")
        return df.interpolate(method=method, axis=axis, limit=limit, limit_area=limit_area, limit_direction=limit_direction)
    else:
        print("There are non-numeric columns")
        return_df = df.copy()
        numeric_cols = list(return_df._get_numeric_data().columns)
        return_df[numeric_cols] = return_df[numeric_cols].interpolate(method=method, axis=axis, limit=limit, limit_area=limit_area, limit_direction=limit_direction)
        return return_df

def working_day_process(df, date_col_name, weekday_col_name, working_day_col_name, hour_col_name, month_col_name, week_col_name, non_working_day_list):
    df_copy = df.copy()
    df_copy[weekday_col_name] = df_copy[date_col_name].dt.weekday
    df_copy["jour"] = df_copy[date_col_name].dt.day
    df_copy["minute"] = df_copy[date_col_name].dt.minute
    df_copy[working_day_col_name] = ~df_copy[weekday_col_name].isin(non_working_day_list)
    df_copy[hour_col_name] = df_copy[date_col_name].dt.hour
    df_copy[month_col_name] = df_copy[date_col_name].dt.month
    df_copy[week_col_name] = df_copy[date_col_name].dt.week
    df_copy["annee"] = df_copy[date_col_name].dt.year
    return df_copy

def load_and_clean_firefly(file_path, xlsx_sheet_number, time_column_name, emplacement, file_missing_values):
    df_firefly = pd.read_excel(io=file_path, sheet_name=1, header=1)
    #Suppression des colonnes vides
    df_firefly = df_firefly.loc[:, ~df_firefly.columns.str.contains('^Unnamed')]
    #Suppression des retours a la ligne dans le nom des colonnes
    df_firefly.columns = [col.replace('\n', '') for col in df_firefly.columns]
    #Renommage de la colonne du timestamp
    df_firefly = df_firefly.rename(columns={df_firefly.columns[0]: time_column_name })
    #Imputation des valeurs manquantes
    stats_desc_df(df_firefly, file_path, file_missing_values)
    """
    if check_na_values(df_firefly) == True:
        df_firefly = interpolate_dataframe(df_firefly, "linear", 0, limit=1, limit_area=None)
    """
    #Ajout des jours travailles
    df_firefly = working_day_process(df_firefly, time_column_name, "jour_semaine", "jour_travaille", "heure", "mois", "semaine", [6,7])
    #Ajout de la colonne emplacement
    df_firefly["emplacement"] = emplacement
    df_firefly = df_firefly.set_index("Date")
    
    return df_firefly

def process_sensor_type_firefly(file_path):
    df = pd.read_excel(io=file_path, sheet_name=0, header=4)
    emplacement = df.columns[0].split('\n')[1].split("Emplacement: ")[1]
    return emplacement

def load_clean_concatenate_firefly(parent_folder_path, key_name, xlsx_sheet_number, time_column_name, file_missing_values):
    list_df = []
    for filename in glob.iglob(parent_folder_path + '*', recursive=True):
        
        if ((key_name in filename) & ('~' not in filename)):
            emplacement = process_sensor_type_firefly(filename)
            df = load_and_clean_firefly(filename, xlsx_sheet_number, time_column_name, emplacement, file_missing_values)
            list_df.append(df)
            
    return pd.concat(list_df, ignore_index=False).sort_index("index")

def load_and_clean_ecosmart(file_path, xlsx_sheet_number, time_column_name, file_missing_values):
    
    df_ecosmart = pd.read_excel(io=file_path, sheet_name=xlsx_sheet_number, header=4)
    columns_drop_list = ['Latitude', 'Longitude', "Altitude", "GPS Date", "Firmware Revision", "Communication"]
    columns_drop_list.append(df_ecosmart.columns[0])
    df_ecosmart = df_ecosmart.drop(columns_drop_list, axis=1)
    df_ecosmart = df_ecosmart.rename(columns={df_ecosmart.columns[0]: time_column_name })
    stats_desc_df(df_ecosmart, file_path, file_missing_values)
    """
    if check_na_values(df_ecosmart) == True:
        df_ecosmart = interpolate_dataframe(df_ecosmart, "linear", axis=0, limit=None, limit_area=None, limit_direction="both")
    """
    df_ecosmart["cov(μg/m3)"] = (df_ecosmart["COV"]*56.106)/24.45
    df_ecosmart = working_day_process(df_ecosmart, time_column_name, "jour_semaine", "jour_travaille", "heure", "mois", "semaine", [6,7])
    df_ecosmart = df_ecosmart.set_index(time_column_name)
    
    return df_ecosmart

def load_clean_zaack(file_path, time_column_name, emplacement, file_missing_values):
    df_zaack = pd.read_csv(file_path, sep=";", header=0, infer_datetime_format=True, parse_dates=True).drop(["sensor_id"], axis=1)
    df_zaack = df_zaack.rename(columns={df_zaack.columns[-1]: time_column_name })
    df_zaack = df_zaack.replace(",", ".", regex=True)
    columns = list(df_zaack.columns)
    types_list = [np.float32, np.float32, np.float32, 
                  np.float32,np.float32,np.int,np.float32,np.float32, 
                  np.int, np.float32, np.float32, datetime.datetime]
    types_dict = dict(zip(columns,types_list))
    df_zaack = df_zaack.astype(types_dict)
    df_zaack["cov(μg/m3)"] = (df_zaack["cov"]*56.106*1000)/24.45
    stats_desc_df(df_zaack, file_path, file_missing_values)
    df_zaack[time_column_name]=pd.to_datetime(df_zaack[time_column_name], format="%d/%m/%Y %H:%M:%S", exact=False)
    df_zaack = working_day_process(df_zaack, time_column_name, "jour_semaine", "jour_travaille", "heure", "mois", "semaine", [6,7])
    df_zaack["seconde"] = df_zaack[time_column_name].dt.second
    df_zaack["emplacement"] = emplacement
    df_zaack = df_zaack.set_index(time_column_name)
    
    return df_zaack

def load_clean_concatenate_zaack(parent_folder_path, key_name, file_missing_values):
    list_df = []
    for filename in glob.iglob(parent_folder_path + '*', recursive=True):
        
        if ((key_name in filename) & ('~' not in filename)):
            emplacement = filename.split('_')[-1].split('.')[0]
            df = load_clean_zaack(filename, "Date", emplacement, file_missing_values)
            list_df.append(df)
            
    return pd.concat(list_df, ignore_index=False).sort_index("index")

def open_file(directory, file_name):
    path = directory+"/"+file_name
    if not os.path.exists(directory):
        os.makedirs(directory)
    if os.path.exists(path):
        f = open(path, "r+")
    else:
        f = open(path, "w+")
    return f

def comparison_firefly_zaack(df_firefly, indic_firefly, df_zaack, indic_zaack ,start, end, moyenne=False):
    df_f = df_firefly.loc[(df_firefly.index <= end) & (df_firefly.index >= start)]
    df_z = df_zaack.loc[(df_zaack.index <= end) & (df_zaack.index >= start)]

    df_f = df_f[[indic_firefly]]
    df_f = df_f.groupby(pd.Grouper(freq='10Min')).aggregate(np.mean)
    df_f[indic_firefly] = df_f[indic_firefly].replace({0:np.nan})
    
    if moyenne == True:
        df_f = df_f.groupby([df_f.index.hour, df_f.index.minute]).mean()
        df_f["hour"] = df_f.index.get_level_values(0)
        df_f["minute"] = df_f.index.get_level_values(1)
        df_f["month"] = 1
        df_f["year"] = 2018
        df_f["day"] = 1
        df_f["Date"] = pd.to_datetime(df_f[['year', 'month', 'day', 'hour', 'minute']])
        df_f['Date'] = [datetime.datetime.time(d) for d in df_f['Date']]
        df_f.index = df_f["Date"]
        df_f = df_f.drop(["hour", "minute", "month", "year", "day", "Date"], axis=1)
   
    df_z = df_z[[indic_zaack]]
    df_z = df_z.groupby(pd.Grouper(freq='10Min')).aggregate(np.mean)
    df_z[indic_zaack] = df_z[indic_zaack].replace({0:np.nan})
    
    if moyenne == True:
        df_z = df_z.groupby([df_z.index.hour, df_z.index.minute]).mean()
        df_z["hour"] = df_z.index.get_level_values(0)
        df_z["minute"] = df_z.index.get_level_values(1)
        df_z["month"] = 1
        df_z["year"] = 2018
        df_z["day"] = 1
        df_z["Date"] = pd.to_datetime(df_z[['year', 'month', 'day', 'hour', 'minute']])
        df_z['Date'] = [datetime.datetime.time(d) for d in df_z['Date']]
        df_z.index = df_z["Date"]
        df_z = df_z.drop(["hour", "minute", "month", "year", "day", "Date"], axis=1)
    
    df_merge = pd.merge(df_f, df_z, left_index=True, right_index=True, how="outer")
    
    return df_merge

def print_comparison_capteur(df_merge, indicateur_1, indicateur_2, title, sub_title1, sub_title2, xlabel, ylabel):
    ax = df_merge[[indicateur_1, indicateur_2]].plot(use_index=True, 
                                                               subplots=True,
                                                               title=title,
                                                               figsize=(18,13),
                                                            fontsize=20)

    ax[0].set_title(sub_title1)
    ax[0].set_ylabel(ylabel, fontsize=20)
    ax[0].set_xlabel(xlabel, fontsize=20)
    ax[0].title.set_size(20)
    ax[1].set_title(sub_title2)
    ax[1].set_ylabel(ylabel, fontsize=20)
    ax[1].set_xlabel(xlabel, fontsize=20)
    ax[1].title.set_size(20)
    
    # set monthly locator
    #ax[0].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    # set formatter
    #ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%a-%d-%m'))
    # set font and rotation for date tick labels
    plt.gcf().autofmt_xdate()

def resolve_dataframes_for_printing(list_tuples, list_dates, moyenne=False, replace_zero=False):
    new_tup_list = []
    for tup in list_tuples:
        tmp = tup[0]
        for idx, date in enumerate(list_dates):
            #Start
            if idx%2 == 0:
                tmp = tmp.loc[(tmp.index >= date)]
            #End
            else:
                tmp = tmp.loc[(tmp.index <= date)]
        tmp = tmp[tup[1]]
        if replace_zero == True:
            dict_replace = (dict(((s,{0: np.nan}) for s in tup[1])))
            tmp = tmp.replace(dict_replace)

        tmp = tmp.groupby(pd.Grouper(freq='10Min')).aggregate(np.mean)

        window = tup[3]
        if window != None:
	        for col in tup[1]:
	        	tmp[col] = tmp.rolling(window=window).mean()[col]

        if moyenne == True:
            tmp = tmp.groupby([tmp.index.hour, tmp.index.minute]).mean()
            tmp["hour"] = tmp.index.get_level_values(0)
            tmp["minute"] = tmp.index.get_level_values(1)
            tmp["month"] = 1
            tmp["year"] = 2018
            tmp["day"] = 1
            tmp["Date"] = pd.to_datetime(tmp[['year', 'month', 'day', 'hour', 'minute']])
            tmp['Date'] = [datetime.datetime.time(d) for d in tmp['Date']]
            tmp.index = tmp["Date"]
            tmp = tmp.drop(["hour", "minute", "month", "year", "day", "Date"], axis=1)
        new_tup_list.append((tmp, tup[1], tup[2]))
    return new_tup_list

def afficher_graphes(liste_tuples, 
                     liste_dates,
                     title,
                     moyenne=False, 
                     remplacer_zero=False, 
                     afficher_capteur_verticalement=True,
                     mode="lines"):

    tuples_to_print = resolve_dataframes_for_printing(liste_tuples, liste_dates, moyenne=moyenne, replace_zero=remplacer_zero)
    list_df, columns, names = zip(*tuples_to_print)
    courbes=[]
    
    max_columns=0
    for j in columns:
        if len(j)> max_columns:
            max_columns=len(j)
    
    if afficher_capteur_verticalement == True:
        fig = tools.make_subplots(rows=max_columns, cols=len(list_df))
    else:
        fig = tools.make_subplots(rows=len(list_df), cols=max_columns)
        
    for idx, capteur in enumerate(list_df):
        for idx_column, column in enumerate(columns[idx]):
            name=column+' ('+names[idx]+')'
            trace = go.Scatter(x=capteur.index, y=capteur[column], name=name, connectgaps=False, mode=mode)
            if afficher_capteur_verticalement == True:
                x=idx_column+1
                y=idx+1
                position_axe = len(list_df)*(idx_column)+(idx+1)
            else:
                x=idx+1
                y=idx_column+1
                position_axe = idx*max_columns+(idx_column+1)
            fig.append_trace(trace, x, y)
            
            fig['layout']['xaxis'+str(position_axe)].update(title='Temps')
            fig['layout']['yaxis'+str(position_axe)].update(title=column)
            if moyenne == True:
                fig['layout']['xaxis'+str(position_axe)].update(tickangle=52)
    
    fig["layout"].update(title=title)
    fig['layout'].update(height=800, width=1000)
    iplot(fig) 