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