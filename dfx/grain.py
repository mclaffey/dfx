import itertools

# needed for find_missing
import pandas as pd

class GrainException(Exception):
    pass

class GrainColumn(object):
    """Provide convenience functions for each column in the GrainDf
    class

    my_df.my_col.filter_up('some value')

    """
    def __init__(self, col_name, grain_df):
        self._col_name = col_name
        self._gdf = grain_df

    def filter_up(self, value):
        """#TODO I don't have an intuitive explanation for this.

        """
        return self._grain_df._df

    def filter_up(self, val, also_above=[]):
        """Procedure: Filter to rows that have the given value in this
        column. Take the ID values for these rows, and find all the other
        rows with these ID values.    

        :val - number, str, None, other. The value to search for in
               the given column. If None, searches for null values.

        :also_above - list of strings, strings are column names, default
               empty. Columns to ignore when filtering, to return more
               results.

        """

        # find rows with the target value
        if val is not None:
            target_i = self._gdf._df[self._col_name] == val
        else:
            target_i = self._gdf._df[self._col_name].isnull()
        target_df = self._gdf._df[target_i].copy()

        # if also_above is a string, convert to list of 1
        if isinstance(also_above, str):
            also_above = [also_above]

        # exception if any of the also_above columns aren't in df
        bad_above = [col for col in also_above
                     if col not in self._gdf.columns]
        if bad_above:
            raise ValueError("Column(s) not a grain column",
                             bad_above, self._gdf.columns)

        # determine which columns to look at for filter values
        other_cols = [col for col in self._gdf.columns
                      if col != self._col_name \
                      and col not in also_above]

        # for target rows, find values in the other grain columns
        other_vals = target_df.groupby(other_cols).\
                     size().to_frame().reset_index().iloc[:, :-1]

        # inner join back to main dataframe to filter on values in
        # other grain columns
        res_df = other_vals.merge(self._gdf._df)

        return res_df
    

class GrainDf(object):
    """Identify and manipulate the grain of a DataFrame
    """
    def __init__(self, df,
                 columns=None,
                 force=False,
                 uniq_threshold=1):
        """Parse a dataframe for grain information

        : df - Pandas dataframe to analyze
        
        :columns - list of strings, each string a column name. This is
                   the list that ._find_grain() will search to find
                   the grain columns. If 'force' is True, this this is
                   used as is for the grain columns, even if not
                   unique.

        :force - boolean, default False. If False, _find_grain() will
                   search 'columns' combinations to find those that
                   uniquely identify rows. If force is True, the
                   'columns' argument will be used the grain columns
                   without searching.

        :uniq_threshold - float, 0-1, default 1. The minimum percent
                   of rows to be uniquely identified when
                   searching. If 1, the grain columns must uniquely
                   identify all rows.

        """
        if df is None:
            raise ValueError("Must provide df argument")
        self._df = df

        # columns should be list. If string, convert to list
        if isinstance(columns, str):
            columns = [columns]
        if columns is not None:
            bad_cols = [col for col in columns if col not in df.columns]
            if bad_cols:
                raise KeyError("Columns not in df", bad_cols)


        if force:
            # grain columns are specified by user
            self.columns = columns
            self.coverage_rate = -1            
        else:
            # search for grain columns
            if uniq_threshold==1:
                # return the first combo that unique identifies
                col_combo_list = self._find_grain(
                    max_combos=1,
                    uniq_threshold=1,
                    columns=columns)
            else:
                col_combo_list = self._find_grain(
                    max_combos=None,
                    uniq_threshold=uniq_threshold,
                    columns=columns)
                
            if not force and not col_combo_list:
                raise GrainException(f"No unique cols. " +\
                                     f"Threshold {uniq_threshold}")
                
            self.columns = col_combo_list[0][0]
            self.coverage_rate = col_combo_list[0][1]

        # add attribute for each column
        for col in self.columns:
            if not hasattr(self, col):
                setattr(self, col, GrainColumn(col, self))

        # calculate on demand
        self._unique_rate = None

    def __str__(self):
        return \
            f"Unique {self.unique_rate:.0%} " + \
            f"Missing {self.missing_rate:.0%} " + \
            str(self.contrib)

    @property
    def summary(self):
        if not hasattr(self, '_summary'):
            # single column, single line summary
            if len(self.columns) == 1:
                self._summary = f"Unique {self.unique_rate:.0%}, " +\
                                self.columns[0]
                return self._summary

            # multi-column summary
            s = []
            s.append(f"Unique {self.unique_rate:.0%}")
            for col, contrib in self.contrib.items():
                s.append(f"  {contrib:4.0%} {col:}")
            for col1, col2, rel in self.col_rels():
                s.append(f"     {col1:20s} {col2:20s} {rel}")
            self._summary = "\n".join(s)
        return self._summary

    def pprint(self):
        print(self.summary)

    @property
    def unique_rate(self):
        if self._unique_rate is None:
            rc = self._df.groupby(list(self.columns)).size()
            self._unique_rate = rc[rc==1].size / self._df.shape[0]
        return self._unique_rate            

    @property
    def unique(self):
        return self.unique_rate == 1
    
    @property
    def contrib(self):
        """Measures the relative contribution of each grain column

        Returns a dict whose keys are str column names and whose
        values are the columns' contribution as a float between >0 -
        1.

        The higher the contribution, the more unique values the column
        is contributing to the total unique combinations across all
        columns.  The sum on contributions across all columns will be
        1. A column will only have 1.0 by itself if it is the only
        grain column. A column can only contrib 0.0 if it has a single
        value across all rows and has been manually forced to be
        included as a grain column. Order of the grain columns does
        not matter.

        The process for calculating contribution is:
        
          1. for each column, find the mode

          2. for each column, count the number of rows that use
          a value other than the mode
          (e.g. employee_id = 9, status = 1)

          3. add up all the values
          (e.g. total = 10)

          4. divide each column's non-mode count by the total
          (e.g. employee_id = 9/10=0.9, status=1/10=0.1)

        """
        non_mode_n = {}
        for col_name in self.columns:
            mode = self._df[col_name].mode(dropna=False)[0]
            # if mode doesn't equal itself, it's pd.nan
            if mode is None or mode != mode:
                non_mode_n[col_name] = (~self._df[col_name]\
                                        .isnull()).sum()
            else:
                non_mode_n[col_name] = (self._df[col_name]\
                                        !=mode).sum()
        non_mode_total = sum(non_mode_n.values())
        contrib = {col_name: non_mode_n[col_name] / non_mode_total
                   for col_name in self.columns}
        return contrib
            
    def _find_grain(self, max_combos, uniq_threshold, columns=None):
        """Find column, or combination of columns, that uniquely identify
        rows

        Returns a list of (col_combo, uniq_rate)
          col_combo is a tuple of str column names
          uniq_rate is a float of how many rows have unique values
            for the given col_combo, 0-1

        This method does not return redundant combinations that aren't
        any better than their parts. For example, if some_id is .95
        unique, and the combination (some_id, some_constant) is also
        .95, the combination of some_id/some_constant is not
        considered.

        :max_combos, int. Stop looking after this many field
          combinations are found. By default, stop once a single combo
          is found that meets the threshold. If max_combos is None,
          there is no limit and this will exhaustively check all field
          combinations.

          max_combos is not performing a top N. In other words, this
          method is NOT exhaustively examining all possible
          combinations of columns, sorting on their unique rate, and
          then returning the top N results. It examines field
          combinations in order, starting with 1-field combinations,
          then 2-field combinations, etc. As soon as it finds
          `max_combo` number of solutions, it returns. It is possible
          (usually likely) that subsequent combinations using more
          columns will have higher unique rates, but that's not how
          the method is intended to search. If you want that result,
          submit with max_combos=None and then sort the returned
          results from the calling code.
        
        :uniq_threshold, float, 0-1. For a field combination to be
          returned, it must uniquely identify this portion of rows
          (e.g. .8 means a combination must uniquely identify at least
          80% of rows)

        :columns, iterable of strings. If provided, consider only
          these columns as candidates for grain. If not provided, all
          columns are considered. This is useful for skipping known
          measure or garbage fields, or paring down the candidates on
          a wide table.

        """
        grain_cols = []
        df = self._df
        if columns is None:
            columns=df.columns
        # start with 1 column at a time, then combinations of 2 columns,
        # increases in size until we have to look at all columns
        for col_n in range(1, len(columns)+1):
            if max_combos is not None and len(grain_cols)>=max_combos:        
                break
            for col_combo in itertools.combinations(columns, col_n):

                # rc = row counts
                rc = df.groupby(list(col_combo)).size()
                uniq_rate = rc.size / df.shape[0]

                # skip if not unique enough
                if uniq_rate < uniq_threshold:
                    continue

                # if combination (col_1, col_2, col_3) isn't any
                # better than, for example, col_1 on it's own, or
                # (col_2, col_3), don't report the larger combination
                ex_is_better = False
                for ex_combo, ex_uniq in grain_cols:
                    if not set(ex_combo).difference(set(col_combo)):
                        if ex_uniq == uniq_rate:
                            ex_is_better = True
                            break

                if not ex_is_better:
                    grain_cols.append((col_combo, uniq_rate))
                    if max_combos is not None \
                       and len(grain_cols) >= max_combos:
                        break

        # sort
        sort_on_uniq=lambda x: x[1]
        return sorted(grain_cols, reverse=True, key=sort_on_uniq)

    def _count_rows(self,
                    columns=None,
                    sort='counts',
                    label='row_count'):
        """For each value combination in columns, show number of rows

        Returns a dataframe, whose first column is the row count
        and the remaining columns are from the columns argument

        Helper function behind .row_counts()

        arguments
        ----------
        
        columns - list of strings to include in row counts. Default
                  None.  If None, uses the grain columns (.columns)
        
        sort - str: 'counts', 'values'. Default 'counts'.  If
                  'counts', sorts with highest row counts first, then
                  on values. If 'values', sorts only on values.
        
        label - str, default 'row_count'. Name for the row count
                  column in the output dataframe.

        """
        if columns is None:
            columns = list(self.columns)

        rc = self._df.groupby(columns).size().reset_index()

        # rename row_count col and move to beginning
        rc.rename(columns={0: label}, inplace=True)
        rc = rc[ [label] + [col for col in rc.columns
                            if col != label]]

        # sort
        sort_asc = [True] * rc.shape[1]
        sort_by = list(rc.columns)
        if sort=='counts':
            # sort on counts first, descending, then all other columns
            sort_asc[0] = False
        elif sort=='values':
            # sort first on all columns besides count, then count, all
            # ascending
            sort_by = sort_by[1:] + [label]
        else:
            raise ValueError("Unrecognized value for sort", sort)
        rc = rc.sort_values(by=sort_by, ascending=sort_asc)

        return rc

    @property
    def row_counts(self):
        """Show number of rows for each value combination in grain
        columns

        Returns a dataframe where first column is row counts and
        remaining columns are from .columns

        """
        if not hasattr(self, '_row_counts') or \
           self._row_counts is None:
            self._row_counts = self._count_rows()
        return self._row_counts    

    def _find_missing(self):
        """Helper function to .missing_rows, see docs there.

        Returns a tuple (missing_df, missing_rate)
        
        """
        # build all_df, the full cartesian cross of all grain values
        col_uniqs = [self._df[col].unique() for col in self.columns]
        all_df = pd.DataFrame(
            data = itertools.product(*col_uniqs)
            ,columns = self.columns)
        all_n = all_df.shape[0]

        # left join all_df to unique value combinations, keep records that
        # are only on the left, return result
        joined_df = all_df.merge(self.row_counts, how='left', indicator=True)
        missing_df = joined_df[joined_df._merge=='left_only']
        missing_df = missing_df.drop(labels=['row_count', '_merge'],
                                     axis=1)

        missing_rate = missing_df.shape[0] / all_n

        return missing_df, missing_rate

    @property
    def missing_rows(self):
        """Find value combinations that aren't represented in the
        data.

        Returns a dataframe with same columns as .columns
        and one row for each value combination that isn't in the data.
        
        """
        if not hasattr(self, '_missing_rows') or\
           self._missing_rows is None:
            self._missing_rows, self._missing_rate\
                = self._find_missing()
        return self._missing_rows

    @property
    def missing_rate(self):
        """Returns the percent of rows that are missing from
        perfect
        """
        # call missing_rows to make sure _missing_rate has been
        # calculated and assigned
        self.missing_rows
        
        return self._missing_rate
    
    @property
    def perfect(self):
        """True if there is 1 row for every grain value combination

        Returns a boolean indicating if the dataframe is perfect.
        
        This takes unique combinations of each value from each grain
        column (i.e. cartesian product), and checks if one and one row
        exists for that unique combination.

        A dataframe is not perfect if:
          - it's not unique (the grain columns don't uniquely identify
            each row), or
          - it is missing combinations

        """
        if not self.unique:
            return False

        return self.missing_rows.empty

        # # build perf_combos, the full cartesian cross of all grain values
        # col_uniqs = [self._df[col].unique() for col in self.columns]
        # perf_combos = itertools.product(*col_uniqs)

        # # calculate lengths
        # perf_n = sum([1 for _ in perf_combos])
        # act_n = self.row_counts.shape[0]
        # is_perf = perf_n == act_n

        # #print("Perfect?", is_perf, perf_n, act_n)
        # return is_perf

    
    def duplicate_ids(self):

        """Return ID values that appear in more than one row
    
        Returns a sorted dataframe with one column for each grain
        column, one row for each value combination, and sorted on a
        'row_count' column which indicates how many rows in the
        dataframe have those values.
        """

        rc = self.row_counts
        rc = rc[rc.row_count>1].copy()
        return rc

    def duplicate_rows(self):
        """Return rows from original dataframe that have duplicate
        values in the grain columns
        
        """
        return self.duplicate_ids().merge(self._df)

    def _get_col_rel(self, col_1, col_2):        
        """Determine if 2 columns are 1:1, 1:many, or many:many

        Helper function to .col_rels()

        Returns a string:
           '1:1'
           '1:many'
           'many:1'
           'many:many'         
        
        paramters
        ---------
        col_1, col_2. str. column names.
        
        """
        if col_1 not in self._df:
            raise KeyError("col_1 not in dataframe",
                             col_1, self._df.columns)
        if col_2 not in self._df:
            raise KeyError("col_2 not in dataframe",
                             col_2, self._df.columns)

        
        # get unique combinations of the two fields (the row counts
        # aren't used)
        cols = self._df.groupby([col_1, col_2])\
                       .size().to_frame().reset_index()

        # determine which side(s) reduced
        uniq_1 = cols.iloc[:,0].unique().size
        reduced_1 = uniq_1 < cols.shape[0]
        uniq_2 = cols.iloc[:,1].unique().size
        reduced_2 = uniq_2 < cols.shape[0]

        if reduced_1 & reduced_2:
            s = 'many:many'
        elif reduced_1 & ~reduced_2:
            s = '1:many'
        elif ~reduced_1 & reduced_2:
            s = 'many:1'
        else:
            s = '1:1'

        return s

    
    
    def col_rels(self):
        """For each pair of grain columsn, determines if the 2 columns are
        1:1, 1:many, or many:many

        Returns a list of tuples, where each tuple is
           (col_1_name, col_2_name, rel)

           rel is a string which is one of these values:
              '1:1'
              '1:many'
              'many:1'
              'many:many'

        If there is only one grain column, returns:
           [(col_1_name, None, 'only-column')]

        """
        if len(self.columns) < 2:
            return [(self.columns[0], None, 'only-column')]    
        col_rels = []
        for col_1, col_2 in itertools.combinations(self.columns, 2):
            col_rels.append((col_1, col_2, self._get_col_rel(col_1, col_2)))
        return col_rels
