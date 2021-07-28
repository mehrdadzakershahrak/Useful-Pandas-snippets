# Useful pandas snippet

## The pandas DataFrame Object

### Start by importing these Python modules
These are recommended import aliases

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from pandas import DataFrame, Series

### The conceptual model
__DataFrame object:__ The pandas DataFrame is a twodimensional table of data with column and row indexes.
The columns are made up of pandas Series objects.

__Series object:__ an ordered, one-dimensional array of data with an index. All the data in a Series is of the same data type. Series arithmetic is vectorised after first aligning the Series index for each of the operands.

    s1 = Series(range(0,4)) # -> 0, 1, 2, 3
    s2 = Series(range(1,5)) # -> 1, 2, 3, 4
    s3 = s1 + s2 # -> 1, 3, 5, 7
    s4 = Series(['a','b'])*3 # -> 'aaa','bbb'

__The index object:__ The pandas Index provides the axis labels for the Series and DataFrame objects. It can only contain hashable objects. A pandas Series has one Index; and a DataFrame has two Indexes.

    # --- get Index from Series and DataFrame
    idx = s.index
    idx = df.columns # the column index
    idx = df.index # the row index
    # --- some Index attributes
    b = idx.is_monotonic_decreasing
    b = idx.is_monotonic_increasing
    b = idx.has_duplicates
    i = idx.nlevels # multi-level indexes
    # --- some Index methods
    a = idx.values() # get as numpy array
    l = idx.tolist() # get as a python list
    idx = idx.astype(dtype)# change data type
    b = idx.equals(o) # check for equality
    idx = idx.union(o) # union of two indexes
    i = idx.nunique() # number unique labels
    label = idx.min() # minimum label
    label = idx.max() # maximum label

### Get your data into a DataFrame

__Load a DataFrame from a CSV file__

    df = pd.read_csv('file.csv')# often works
    df = pd.read_csv(‘file.csv’, header=0,
    index_col=0, quotechar=’”’,sep=’:’,
    na_values = [‘na’, ‘-‘, ‘.’, ‘’])

Please refer to pandas docs for all arguments

__From inline CSV text to a DataFrame__

    from StringIO import StringIO # python2.7
    #from io import StringIO # python 3
    data = """, Animal, Cuteness, Desirable
    row-1, dog, 8.7, True
    row-2, bat, 2.6, False"""
    df = pd.read_csv(StringIO(data),
    header=0, index_col=0,
    skipinitialspace=True)

Note: _skipinitialspace_=True allows a pretty layout

__Load DataFrames from a Microsoft Excel file__

### Each Excel sheet in a Python dictionary

    workbook = pd.ExcelFile('file.xlsx')
    dictionary = {}
    for sheet_name in workbook.sheet_names:
    df = workbook.parse(sheet_name)
    dictionary[sheet_name] = df

Note: the parse() method takes many arguments like read_csv() above. Refer to the pandas documentation.

__Load a DataFrame from a MySQL database__
    import pymysql
    from sqlalchemy import create_engine
    engine = create_engine('mysql+pymysql://'
    +'USER:PASSWORD@localhost/DATABASE')
    df = pd.read_sql_table('table', engine)

__Data in Series then combine into a DataFrame__

    # Example 1 ...
    s1 = Series(range(6))
    s2 = s1 * s1
    s2.index = s2.index + 2# misalign indexes
    df = pd.concat([s1, s2], axis=1)
    # Example 2 ...
    s3 = Series({'Tom':1, 'Dick':4, 'Har':9})
    s4 = Series({'Tom':3, 'Dick':2, 'Mar':5})
    df = pd.concat({'A':s3, 'B':s4 }, axis=1)

Note: 1st method has in integer column labels
Note: 2nd method does not guarantee col order
Note: index alignment on DataFrame creation

__Get a DataFrame from data in a Python dictionary__
    # default --- assume data is in columns
    df = DataFrame({
    'col0' : [1.0, 2.0, 3.0, 4.0],
    'col1' : [100, 200, 300, 400]
    })
