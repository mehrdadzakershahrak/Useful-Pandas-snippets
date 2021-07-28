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

    #Each Excel sheet in a Python dictionary
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

__Get a DataFrame from data in a Python dictionary__

    # --- use helper method for data in rows
    df = DataFrame.from_dict({ # data by row
    'row0' : {'col0':0, 'col1':'A'},
    'row1' : {'col0':1, 'col1':'B'}
    }, orient='index')
    df = DataFrame.from_dict({ # data by row
    'row0' : [1, 1+1j, 'A'],
    'row1' : [2, 2+2j, 'B']
    }, orient='index')

__Create play/fake data (useful for testing)__

    # --- simple
    df = DataFrame(np.random.rand(50,5))
    # --- with a time-stamp row index:
    df = DataFrame(np.random.rand(500,5))
    df.index = pd.date_range('1/1/2006',
    periods=len(df), freq='M')
    # --- with alphabetic row and col indexes
    import string
    import random
    r = 52 # note: min r is 1; max r is 52
    c = 5
    df = DataFrame(np.random.randn(r, c),
    columns = ['col'+str(i) for i in
    range(c)],
    index = list((string.uppercase +
    string.lowercase)[0:r]))
    df['group'] = list(
    ''.join(random.choice('abcd')
    for _ in range(r))
    )

### Saving a DataFrame

__Saving a DataFrame to a CSV file__

    df.to_csv('name.csv', encoding='utf-8')

__Saving DataFrames to an Excel Workbook__

    from pandas import ExcelWriter
    writer = ExcelWriter('filename.xlsx')
    df1.to_excel(writer,'Sheet1')
    df2.to_excel(writer,'Sheet2')
    writer.save()

__Saving a DataFrame to MySQL__

    import pymysql
    from sqlalchemy import create_engine
    e = create_engine('mysql+pymysql://' +
    'USER:PASSWORD@localhost/DATABASE')
    df.to_sql('TABLE',e, if_exists='replace')

Note: if_exists ! 'fail', 'replace', 'append'

__Saving a DataFrame to a Python dictionary__

    dictionary = df.to_dict()
    Saving a DataFrame to a Python string
    string = df.to_string()

Note: sometimes may be useful for debugging

### Working with the whole DataFrame
__Peek at the DataFrame contents__

    df.info() # index & data types
    n = 4
    dfh = df.head(n) # get first n rows
    dft = df.tail(n) # get last n rows
    dfs = df.describe() # summary stats cols
    top_left_corner_df = df.iloc[:5, :5]

__DataFrame non-indexing attributes__

    dfT = df.T # transpose rows and cols
    l = df.axes # list row and col indexes
    (r, c) = df.axes # from above
    s = df.dtypes # Series column data types
    b = df.empty # True for empty DataFrame
    i = df.ndim # number of axes (2)
    t = df.shape # (row-count, column-count)
    (r, c) = df.shape # from above
    i = df.size # row-count * column-count
    a = df.values # get a numpy array for df

__DataFrame utility methods__

    dfc = df.copy() # copy a DataFrame
    dfr = df.rank() # rank each col (default)
    dfs = df.sort() # sort each col (default)
    dfc = df.astype(dtype) # type conversion

__DataFrame iteration methods__

    df.iteritems()# (col-index, Series) pairs
    df.iterrows() # (row-index, Series) pairs
    # example ... iterating over columns
    for (name, series) in df.iteritems():
    print('Col name: ' + str(name))
    print('First value: ' +
    str(series.iat[0]) + '\n')

__Maths on the whole DataFrame (not a complete list)__

    df = df.abs() # absolute values
    df = df.add(o) # add df, Series or value
    s = df.count() # non NA/null values
    df = df.cummax() # (cols default axis)
    df = df.cummin() # (cols default axis)
    df = df.cumsum() # (cols default axis)
    df = df.cumprod() # (cols default axis)
    df = df.diff() # 1st diff (col def axis)
    df = df.div(o) # div by df, Series, value
    df = df.dot(o) # matrix dot product
    s = df.max() # max of axis (col def)
    s = df.mean() # mean (col default axis)
    s = df.median()# median (col default)
    s = df.min() # min of axis (col def)
    df = df.mul(o) # mul by df Series val
    s = df.sum() # sum axis (cols default)

Note: The methods that return a series default to working on columns.

__DataFrame filter/select rows or cols on label info__

    df = df.filter(items=['a', 'b']) # by col
    df = df.filter(items=[5], axis=0) #by row
    df = df.filter(like='x') # keep x in col
    df = df.filter(regex='x') # regex in col
    df = df.select(crit=(lambda x:not x%5))#r

Note: select takes a Boolean function, for cols: axis=1

Note: filter defaults to cols; select defaults to rows 