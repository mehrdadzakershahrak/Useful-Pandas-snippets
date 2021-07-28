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

Note: if_exists --> 'fail', 'replace', 'append'

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

### Working with Columns

A DataFrame column is a pandas Series object.

__Get column index and labels__

    idx = df.columns # get col index
    label = df.columns[0] # 1st col label
    lst = df.columns.tolist() # get as a list

__Change column labels__

    df.rename(columns={'old':'new'},
    inplace=True)
    df = df.rename(columns={'a':1,'b':'x'})

__Selecting columns__

    s = df['colName'] # select col to Series
    df = df[['colName']] # select col to df
    df = df[['a','b']] # select 2 or more
    df = df[['c','a','b']]# change order
    s = df[df.columns[0]] # select by number
    df = df[df.columns[[0, 3, 4]] # by number
    s = df.pop('c') # get col & drop from df

__Selecting columns with Python attributes__

    s = df.a # same as s = df['a']
    # cannot create new columns by attribute
    df.existing_col = df.a / df.b
    df['new_col'] = df.a / df.b

Trap: column names must be valid identifiers.

__Adding new columns to a DataFrame__

    df['new_col'] = range(len(df))
    df['new_col'] = np.repeat(np.nan,len(df))
    df['random'] = np.random.rand(len(df))
    df['index_as_col'] = df.index
    df1[['b','c']] = df2[['e','f']]
    df3 = df1.append(other=df2)

Trap: When adding an indexed pandas object as a new column, only items from the new series that have a corresponding index in the DataFrame will be added.
The receiving DataFrame is not extended to accommodate the new series. To merge, see below. Trap: when adding a python list or numpy array, the column will be added by integer position. 

__Swap column contents – change column order__

    df[['B', 'A']] = df[['A', 'B']]

__Dropping columns (mostly by label)__

    df = df.drop('col1', axis=1)
    df.drop('col1', axis=1, inplace=True)
    df = df.drop(['col1','col2'], axis=1)
    s = df.pop('col') # drops from frame
    del df['col'] # even classic python works
    df.drop(df.columns[0], inplace=True)

__Vectorised arithmetic on columns__

    df['proportion']=df['count']/df['total']
    df['percent'] = df['proportion'] * 100.0

__Apply numpy mathematical functions to columns__

    df['log_data'] = np.log(df['col1'])
    df['rounded'] = np.round(df['col2'], 2)

Note: Many more mathematical functions

__Columns value set based on criteria__

    df['b']=df['a'].where(df['a']>0,other=0)
    df['d']=df['a'].where(df.b!=0,other=df.c)

Note: where other can be a Series or a scalar

__Data type conversions__

    s = df['col'].astype(str) # Series dtype
    na = df['col'].values # numpy array
    pl = df['col'].tolist() # python list

Note: useful dtypes for Series conversion: int, float, str

Trap: index lost in conversion from Series to array or list

__Common column-wide methods/attributes__

    value = df['col'].dtype # type of data
    value = df['col'].size # col dimensions
    value = df['col'].count()# non-NA count
    value = df['col'].sum()
    value = df['col'].prod()
    value = df['col'].min()
    value = df['col'].max()
    value = df['col'].mean()
    value = df['col'].median()
    value = df['col'].cov(df['col2'])
    s = df['col'].describe()
    s = df['col'].value_counts()

__Find index label for min/max values in column__

    label = df['col1'].idxmin()
    label = df['col1'].idxmax()

__Common column element-wise methods__

    s = df['col'].isnull()
    s = df['col'].notnull() # not isnull()
    s = df['col'].astype(float)
    s = df['col'].round(decimals=0)
    s = df['col'].diff(periods=1)
    s = df['col'].shift(periods=1)
    s = df['col'].to_datetime()
    s = df['col'].fillna(0) # replace NaN w 0
    s = df['col'].cumsum()
    s = df['col'].cumprod()
    s = df['col'].pct_change(periods=4)
    s = df['col'].rolling_sum(periods=4,
    window=4)

Note: also rolling_min(), rolling_max(), and many more.

__Append a column of row sums to a DataFrame__

    df['Total'] = df.sum(axis=1)

Note: also means, mins, maxs, etc.

__Multiply every column in DataFrame by Series__

    df = df.mul(s, axis=0) # on matched rows

Note: also add, sub, div, etc.

__Selecting columns with .loc, .iloc and .ix__

    df = df.loc[:, 'col1':'col2'] # inclusive
    df = df.iloc[:, 0:2] # exclusive

__Get the integer position of a column index label__

    j = df.columns.get_loc('col_name')

__Test if column index values are unique/monotonic__

    if df.columns.is_unique: pass # ...
    b = df.columns.is_monotonic_increasing
    b = df.columns.is_monotonic_decreasing

### Working with rows

__Get the row index and labels__

    idx = df.index # get row index
    label = df.index[0] # 1st row label
    lst = df.index.tolist() # get as a list

__Change the (row) index__

    df.index = idx # new ad hoc index
    df.index = range(len(df)) # set with list
    df = df.reset_index() # replace old w new
    # note: old index stored as a col in df
    df = df.reindex(index=range(len(df)))
    df = df.set_index(keys=['r1','r2','etc'])
    df.rename(index={'old':'new'},
    inplace=True)

__Adding rows__

    df = original_df.append(more_rows_in_df)

Hint: convert to a DataFrame and then append. Both
DataFrames should have same column labels.

__Dropping rows (by name)__

    df = df.drop('row_label')
    df = df.drop(['row1','row2']) # multi-row

__Boolean row selection by values in a column__

    df = df[df['col2'] >= 0.0]
    df = df[(df['col3']>=1.0) |
    (df['col1']<0.0)]
    df = df[df['col'].isin([1,2,5,7,11])]
    df = df[~df['col'].isin([1,2,5,7,11])]
    df = df[df['col'].str.contains('hello')]

Trap: bitwise "or", "and" “not” (ie. | & ~) co-opted to be Boolean operators on a Series of Boolean
Trap: need parentheses around comparisons.

__Selecting rows using isin over multiple columns__

    # fake up some data
    data = {1:[1,2,3], 2:[1,4,9], 3:[1,8,27]}
    df = pd.DataFrame(data)
    # multi-column isin
    lf = {1:[1, 3], 3:[8, 27]} # look for
    f = df[df[list(lf)].isin(lf).all(axis=1)]

__Selecting rows using an index__

    idx = df[df['col'] >= 2].index
    print(df.ix[idx])

__Select a slice of rows by integer position__

[inclusive-from : exclusive-to [: step]]
default start is 0; default end is len(df)

    df = df[:] # copy DataFrame
    df = df[0:2] # rows 0 and 1
    df = df[-1:] # the last row
    df = df[2:3] # row 2 (the third row)
    df = df[:-1] # all but the last row
    df = df[::2] # every 2nd row (0 2 ..)

Trap: a single integer without a colon is a column label for integer numbered columns.

__Select a slice of rows by label/index__

[inclusive-from : inclusive–to [ : step]]

    df = df['a':'c'] # rows 'a' through 'c'

Trap: doesn't work on integer labelled rows

__Append a row of column totals to a DataFrame__

    # Option 1: use dictionary comprehension
    sums = {col: df[col].sum() for col in df}
    sums_df = DataFrame(sums,index=['Total'])
    df = df.append(sums_df)
    # Option 2: All done with pandas
    df = df.append(DataFrame(df.sum(),
    columns=['Total']).T)

__Iterating over DataFrame rows__
    for (index, row) in df.iterrows(): # pass

Trap: row data type may be coerced.

__Sorting DataFrame rows values__

    df = df.sort(df.columns[0],
    ascending=False)
    df.sort(['col1', 'col2'], inplace=True)

__Random selection of rows__

    import random as r
    k = 20 # pick a number
    selection = r.sample(range(len(df)), k)
    df_sample = df.iloc[selection, :]

Note: this sample is not sorted

__Sort DataFrame by its row index__

    df.sort_index(inplace=True) # sort by row
    df = df.sort_index(ascending=False)

__Drop duplicates in the row index__

    df['index'] = df.index # 1 create new col
    df = df.drop_duplicates(cols='index',
    take_last=True)# 2 use new col
    del df['index'] # 3 del the col
    df.sort_index(inplace=True)# 4 tidy up

__Test if two DataFrames have same row index__

    len(a)==len(b) and all(a.index==b.index)

__Get the integer position of a row or col index label__

    i = df.index.get_loc('row_label')

Trap: index.get_loc() returns an integer for a unique match. If not a unique match, may return a slice or
mask.

__Get integer position of rows that meet condition__

    a = np.where(df['col'] >= 2) #numpy array

__Test if the row index values are unique/monotonic__

    if df.index.is_unique: pass # ...
    b = df.index.is_monotonic_increasing
    b = df.index.is_monotonic_decreasing

### Working with cells

__Selecting a cell by row and column labels__

    value = df.at['row', 'col']
    value = df.loc['row', 'col']
    value = df['col'].at['row'] # tricky

Note: .at[] fastest label based scalar lookup

__Setting a cell by row and column labels__
    df.at['row, 'col'] = value
    df.loc['row, 'col'] = value
    df['col'].at['row'] = value # tricky

__Selecting and slicing on labels__
    df = df.loc['row1':'row3', 'col1':'col3']

Note: the "to" on this slice is inclusive.

__Setting a cross-section by labels__

    df.loc['A':'C', 'col1':'col3'] = np.nan
    df.loc[1:2,'col1':'col2']=np.zeros((2,2))
    df.loc[1:2,'A':'C']=othr.loc[1:2,'A':'C']

Remember: inclusive "to" in the slice

__Selecting a cell by integer position__

    value = df.iat[9, 3] # [row, col]
    value = df.iloc[0, 0] # [row, col]
    value = df.iloc[len(df)-1,
     len(df.columns)-1]

__Selecting a range of cells by int position__
    
    df = df.iloc[2:4, 2:4] # subset of the df
    df = df.iloc[:5, :5] # top left corner
    s = df.iloc[5, :] # returns row as Series
    df = df.iloc[5:6, :] # returns row as row

Note: exclusive "to" – same as python list slicing.

__Setting cell by integer position__

    df.iloc[0, 0] = value # [row, col]
    df.iat[7, 8] = value

__Setting cell range by integer position__

    df.iloc[0:3, 0:5] = value
    df.iloc[1:3, 1:4] = np.ones((2, 3))
    df.iloc[1:3, 1:4] = np.zeros((2, 3))
    df.iloc[1:3, 1:4] = np.array([[1, 1, 1],
    [2, 2, 2]])

Remember: exclusive-to in the slice

__.ix for mixed label and integer position indexing__

    value = df.ix[5, 'col1']
    df = df.ix[1:5, 'col1':'col3']

__Views and copies__

From the manual: Setting a copy can cause subtle
errors. The rules about when a view on the data is
returned are dependent on NumPy. Whenever an array
of labels or a Boolean vector are involved in the indexing
operation, the result will be a copy.

### In summary: indexes and addresses

In the main, these notes focus on the simple, single level Indexes. Pandas also has a hierarchical or multi-level Indexes (aka the MultiIndex).

A DataFrame has two Indexes
1. Typically, the column index (df.columns) is a list of strings (observed variable names) or (less
commonly) integers (the default is numbered from 0 to length-1)
2. Typically, the row index (df.index) might be:
    1. Integers - for case or row numbers (default is numbered from 0 to length-1);
    2. Strings – for case names; or
    3. DatetimeIndex or PeriodIndex – for time series data (more below)

__Indexing__

    # --- selecting columns
    s = df['col_label'] # scalar
    df = df[['col_label']] # one item list
    df = df[['L1', 'L2']] # many item list
    df = df[index] # pandas Index
    df = df[s] # pandas Series
    # --- selecting rows
    df = df['from':'inc_to']# label slice
    df = df[3:7] # integer slice
    df = df[df['col'] > 0.5]# Boolean Series
    df = df.loc['label'] # single label
    df = df.loc[container] # lab list/Series
    df = df.loc['from':'to']# inclusive slice
    df = df.loc[bs] # Boolean Series
    df = df.iloc[0] # single integer
    df = df.iloc[container] # int list/Series
    df = df.iloc[0:5] # exclusive slice
    df = df.ix[x] # loc then iloc
    # --- select DataFrame cross-section
    # r and c can be scalar, list, slice
    df.loc[r, c] # label accessor (row, col)
    df.iloc[r, c]# integer accessor
    df.ix[r, c] # label access int fallback
    df[c].iloc[r]# chained – also for .loc
    # --- select cell
    # r and c must be label or integer
    df.at[r, c] # fast scalar label accessor
    df.iat[r, c] # fast scalar int accessor
    df[c].iat[r] # chained – also for .at
    # --- indexing methods
    v = df.get_value(r, c) # get by row, col
    df = df.set_value(r,c,v)# set by row, col
    df = df.xs(key, axis) # get cross-section
    df = df.filter(items, like, regex, axis)
    df = df.select(crit, axis)

Note: the indexing attributes (.loc, .iloc, .ix, .at .iat) can be used to get and set values in the DataFrame.

Note: the .loc, iloc and .ix indexing attributes can accept python slice objects. But .at and .iat do not.

Note: .loc can also accept Boolean Series arguments

Avoid: chaining in the form df[col_indexer][row_indexer]

Trap: label slices are inclusive, integer slices exclusive.

### Joining/Combining DataFrames

Three ways to join two DataFrames:
1. merge (a database/SQL-like join operation)
2. concat (stack side by side or one on top of the other)
3. combine_first (splice the two together, choosing values from one over the other)

__Merge on indexes__

    df_new = pd.merge(left=df1, right=df2,
    how='outer', left_index=True,
    right_index=True)

How: 'left', 'right', 'outer', 'inner'

How: outer=union/all; inner=intersection

__Merge on columns__

    df_new = pd.merge(left=df1, right=df2,
    how='left', left_on='col1',
    right_on='col2')

Trap: When joining on columns, the indexes on the passed DataFrames are ignored.

Trap: many-to-many merges on a column can result in an explosion of associated data.

__Join on indexes (another way of merging)__

    df_new = df1.join(other=df2, on='col1',
    how='outer')
    df_new = df1.join(other=df2,on=['a','b'],
    how='outer')

Note: DataFrame.join() joins on indexes by default. DataFrame.merge() joins on common columns by
default. 

__Simple concatenation is often the best__

    df=pd.concat([df1,df2],axis=0)#top/bottom
    df = df1.append([df2, df3]) #top/bottom
    df=pd.concat([df1,df2],axis=1)#left/right

Trap: can end up with duplicate rows or cols

Note: concat has an ignore_index parameter

__Combine_first__

    df = df1.combine_first(other=df2)
    # multi-combine with python reduce()
    df = reduce(lambda x, y:
    x.combine_first(y),
    [df1, df2, df3, df4, df5])

Uses the non-null values from df1. The index of the combined DataFrame will be the union of the indexes
from df1 and df2.

### Groupby: Split-Apply-Combine

The pandas "groupby" mechanism allows us to split the data into groups, apply a function to each group independently and then combine the results.

__Grouping__

    gb = df.groupby('cat') # by one columns
    gb = df.groupby(['c1','c2']) # by 2 cols
    gb = df.groupby(level=0) # multi-index gb
    gb = df.groupby(level=['a','b']) # mi gb
    print(gb.groups)

Note: groupby() returns a pandas groupby object

Note: the groupby object attribute .groups contains a dictionary mapping of the groups.

Trap: NaN values in the group key are automatically dropped – there will never be a NA group.

__Iterating groups – usually not needed__

    for name, group in gb:
        print (name)
        print (group)

__Selecting a group__

    dfa = df.groupby('cat').get_group('a')
    dfb = df.groupby('cat').get_group('b')

__Applying an aggregating function__

    # apply to a column ...
    s = df.groupby('cat')['col1'].sum()
    s = df.groupby('cat')['col1'].agg(np.sum)
    # apply to the every column in DataFrame
    s = df.groupby('cat').agg(np.sum)
    df_summary = df.groupby('cat').describe()
    df_row_1s = df.groupby('cat').head(1)

Note: aggregating functions reduce the dimension by one – they include: mean, sum, size, count, std, var,
sem, describe, first, last, min, max

__Applying multiple aggregating functions__

    gb = df.groupby('cat')
    # apply multiple functions to one column
    dfx = gb['col2'].agg([np.sum, np.mean])
    # apply to multiple fns to multiple cols
    dfy = gb.agg({
    'cat': np.count_nonzero,
    'col1': [np.sum, np.mean, np.std],
    'col2': [np.min, np.max]
    })

Note: gb['col2'] above is shorthand for df.groupby('cat')['col2'], without the need for regrouping.

__Transforming functions__

    # transform to group z-scores, which have
    # a group mean of 0, and a std dev of 1.
    zscore = lambda x: (x-x.mean())/x.std()
    dfz = df.groupby('cat').transform(zscore)
    # replace missing data with group mean
    mean_r = lambda x: x.fillna(x.mean())
    dfm = df.groupby('cat').transform(mean_r)

Note: can apply multiple transforming functions in a manner similar to multiple aggregating functions above.

__Applying filtering functions__

Filtering functions allow you to make selections based on whether each group meets specified criteria

    # select groups with more than 10 members
    eleven = lambda x: (len(x['col1']) >= 11)
    df11 = df.groupby('cat').filter(eleven)
    Group by a row index (non-hierarchical index)
    df = df.set_index(keys='cat')
    s = df.groupby(level=0)['col1'].sum()
    dfg = df.groupby(level=0).sum()

