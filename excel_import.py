# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:00:13 2016

Allow to import excel files to pandas dataframe without issues on excel cells format

@author: fportes
"""

import pandas as pd
import numpy as np
import xlrd, datetime
from collections import OrderedDict

class columns:
    def __init__(self):
        self.given_vars = dict()
        self.ordered_var = OrderedDict()
    def set_text_columns(self, var_list):
        for v in var_list:        
            self.given_vars[v] = column('text')
    def set_numeric_columns(self, var_list):
        for v in var_list:        
            self.given_vars[v] = column('numeric')
    def set_date_columns(self, var_list):
        for v in var_list:        
            self.given_vars[v] = column('datetime')
    def sort(self):
        #create a temp dict with the variable found in excel columns header
        d = {k:v for (k,v) in self.given_vars.items() if self.given_vars[k].in_excel is True}
        #create an ordered dict with the order of excel column index
        self.ordered_var = OrderedDict(sorted(d.items(), key=lambda x: x[1].index))

    def reset(self):
        '''Reset attributes index and in_excel from the given excel columns to parse an other excel'''
        for k in self.given_vars.keys():
            self.given_vars[k].reset_idx()
            self.given_vars[k].reset_in_excel()

class column:
    def __init__(self, dtype):
        self.dtype = dtype
        self.index = None
        self.in_excel = False
    def set_index(self, n):
        if self.index is None:        
            self.index = n
        else:
            self.index = min(self.index, n)
    
    def reset_idx(self):
        self.index = None
    def reset_in_excel(self):
        self.in_excel = False
    def get_index():
        return self.index
    def get_in_excel():
        return self.in_excel        

def convert_to_date(excel_sec, book, missing):
        if excel_sec is not '':
            date = xlrd.xldate_as_tuple(excel_sec, book.datemode)
            return str(datetime.datetime(*date))
        else:
            return missing
def convert_to_num(data, missing):
    if data == '':
        return missing
    else:
        return float(data)
def convert_to_text(data, missing):
    if data == '':
        return missing
    else:
        return str(data)
        
def read_excel(path, cols):
    ''' Read excel file and create a dataframe with the columns order of excel table
    '''
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)

    headers = sheet.row_values(0)
    for idx,col in enumerate(headers):
        cols.given_vars[col].in_excel = True
        cols.given_vars[col].set_index(idx)
    cols.sort()

    df = pd.DataFrame()
    for col, col_attr in cols.ordered_var.items():
        col_values = sheet.col_values(col_attr.index)[1:]
        if col_attr.dtype == 'text':
            d = [convert_to_text(t, np.nan) for t in col_values]
        elif col_attr.dtype == 'datetime':
            d = [convert_to_date(date, book, np.nan) for date in col_values]
        elif col_attr.dtype == 'numeric':
            d = [convert_to_num(f, np.nan) for f in col_values]
        df.loc[:,col] = pd.Series(d)
    cols.reset()
    return df