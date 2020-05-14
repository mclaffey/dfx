import pandas as pd

class ReducedDf(object):
    """Dataframe with single-value columns stored separately
    """
    def __init__(self, df):
        self.nulls = []
        self.zeros = []
        self.constants = {}
        multi_value = []
        for col in df:
            vals = df[col].unique()
            if len(vals) > 1:
                multi_value.append(col)
            elif pd.isnull(vals[0]):
                self.nulls.append(col)
            elif vals == [0]:
                self.zeros.append(col)
            else:
                self.constants[col] = vals[0]
        self.df=df[multi_value].copy()
        
    def __str__(self):
        return f"{self.__class__.__name__} " + \
            f"{self.df.shape[0]} rows, {self.df.shape[1]} cols, " + \
            f"{len(self.constants)} constants, " + \
            f"{len(self.zeros)} zeros, {len(self.nulls)} nulls"

    def _diff(self, other):
        """Return a string indicating difference versus another
        ReducedDf instance, or empty string if no differences.

        used for __eq__
        """
        # type
        if type(other) != type(self):
            return 'Class: {}'.format(type(other))
        # attributes
        attr_diffs = [attr for attr in ['nulls', 'zeros', 'constants']
                      if not getattr(self, attr) == \
                      getattr(other, attr)]
        if attr_diffs:
            return 'Atttributes: {}'.format(attr_diffs)
        # df
        import pandas.testing
        try:
            pandas.testing.assert_frame_equal(self.df, other.df)
        except AssertionError as e:
            return 'Dataframes not equal'

        # otherwise they look equal
        return ''

    def __eq__(self, other):
        return not self._diff(other)

    def pprint(self):
        print(self.summary)

    @property
    def summary(self):
        """Report fields and values
        """
        if not hasattr(self, '_summary'):            
            s = []
            s.append(self.__str__())
            if self.nulls:
                s.append("NULLS: {}".format(" ".join(self.nulls)))
            if self.zeros:
                s.append("Zeros: {}".format(" ".join(self.zeros)))
            if self.constants:            
                s.append("Constants: ")
            for k,v in self.constants.items():
                s.append(f"   {k[:40]}: {str(v)[:80]}")
            s.append("\n" + str(self.df))
            self._summary = "\n".join(s)
        return self._summary
                
class BestReducedDf(object):
    """See if breaking up the dataframe into different row groups
    allows for a more succint summary

    Iterate over each column, and for each value in that column, split
    the rows into groups based on that value. Get the text summary of
    each of those groups, put them together in a mega summary, and
    measure the length of the mega summary in total characters. Create
    that mega summary for column, and pick the column with the
    shortest mega summary.    
    """
    def __init__(self, df):
        
        # Reduce the original dataset
        orig_df = df
        rdf = ReducedDf(orig_df)
        df = rdf.df
        
        # list of columns that original dataset was split on
        self.split_columns = []

        # mega summary of all column value rows, but start with
        # unsplit version to see if any column splits can beat it
        self._summary = rdf.summary
        self._summary_length = len(self._summary)
        self._col_summary_lengths = [
            (None, self._summary_length)]

        # dictionary with keys of column values and values of
        # ReducedDfs for the rows of each value
        self._col_val_rdfs = {}

        # iterate over columns and find the shortest mega summary
        for col_name in df.columns:
            col_val_rdfs = {col_val: ReducedDf(col_val_df)
                            for col_val, col_val_df
                            in df.groupby(col_name)}
            col_val_summaries = []
            for col_val, col_val_rdf in col_val_rdfs.items():
                col_val_summaries.append(f"-- {col_name}={col_val}")
                col_val_summaries.append(col_val_rdf.summary)
            mega_summary = "\n \n".join(col_val_summaries)
            mega_summary_len = len(mega_summary)
            self._col_summary_lengths.append(
                (col_name, mega_summary_len))

            # if this is the shortest, keep it
            if mega_summary_len < self._summary_length:
                self.split_columns = [col_name]
                self._summary = mega_summary
                self._summary_length = mega_summary_len
                self._col_val_rdfs = col_val_rdfs
                
        # for each value in the split column, assign an attribute
        # which points to the ReducedDf of those rows
        for col_val, col_val_rdf in self._col_val_rdfs.items():
            col_val_safe_name = col_val.lower().replace(',.#! ',
                                                        '_____')
            if not hasattr(self, col_val_safe_name):
                try:
                    setattr(self, col_val_safe_name, col_val_rdf)
                except Exception as e:
                    # see what kind of exceptions, probably for
                    # invalid names with punctuation, If it's an easy
                    # fix, handle it, otherwise okay to fail (adding
                    # the attributes are a conveniences and don't
                    # suppoer necessary functions)                    
                    raise e from None

    @property
    def summary(self):
        """Return the mega summary of all column value ReducedDfs
        """
        return self._summary

    def pprint(self):
        """Print summary
        """
        print(self.summary)

class DiffDf(object):
    """Provide row, column and cell differences between two datasets

        # create side-by-side dataframe
        # column order: diff, same, only in 1, only in 2

    """
    def __init__(self, df1, df2):

        # calculate row differences
        not_in_df2_i = df1.index.difference(df2.index)
        self._rows_removed = df1.loc[not_in_df2_i]
        not_in_df1_i = df2.index.difference(df1.index)
        self._rows_added = df2.loc[not_in_df1_i]
        in_both_i = df1.index.intersection(df2.index)
        
        # for common columns, count value difference (pd.Series)
        common_cols = df1.columns.intersection(df2.columns).to_list()
        is_diff_df = df1.loc[in_both_i, common_cols] != \
                  df2.loc[in_both_i, common_cols]
        self.col_diff_counts = is_diff_df.sum() # by column
        self.col_diff_counts.name = 'diff_counts'        
        self.diff_cols    = self.col_diff_counts.index[
            self.col_diff_counts>0].tolist()
        self.no_diff_cols = self.col_diff_counts.index[
            self.col_diff_counts==0].tolist()

        # prepare side-by-side df

        # for df1, rename columns
        # add _1 unless it's a column without any changes
        df1_cols_orig = list(df1.columns)
        df1_cols_new = []
        for col in df1.columns:
            if col in self.no_diff_cols:
                df1_cols_new.append(col)
            else:
                df1_cols_new.append(col + '_1')
        df1 = df1.copy()
        df1.columns = df1_cols_new
        #print(f'df1: {df1.columns}')
        
        # for df2, rename/drop columns
        # add _2 unless its a column without any changes, in which
        # case from from df2 since we'll keep the copy from df1
        df2_cols_orig = list(df2.columns)
        df2 = df2.drop(labels=self.no_diff_cols, axis=1).copy()
        df2.columns = [col + '_2' for col in df2.columns]
        #print(f'df2: {df2.columns}')

        # determine column order in side-by-side
        final_col_order = []
        # first get the diff columns from both dfs
        for col in self.diff_cols:
            final_col_order.append(col + '_1')
            final_col_order.append(col + '_2')
            df1_cols_orig.remove(col)
            df2_cols_orig.remove(col)
        # second get the no diff columns
        for col in self.no_diff_cols:
            final_col_order.append(col)
            df1_cols_orig.remove(col)
            df2_cols_orig.remove(col)
        # third get remaining columns for df1
        final_col_order.extend([col + '_1' for col in df1_cols_orig])
        # fourth get remaining columns for df2
        final_col_order.extend([col + '_2' for col in df2_cols_orig])
        #print(final_col_order)

        # create side-by-side df by merging on index and set new
        # column order
        self.df = df1.merge(df2, left_index=True, right_index=True,
                        how='outer', indicator=True)
        self.df = self.df[['_merge'] + final_col_order]
        
    @property
    def rows_added(self):
        """Returns a dataframe of rows in the second dataframe but not
        in the first.
        """
        return self._rows_added

    @property
    def rows_removed(self):
        """Returns a dataframe of rows in the first dataframe but not
        in the second.
        """
        return self._rows_removed

    def changed(self, columns=[], row_adds=False, row_removes=False):
        """Returns a dataframe of rows with changes in the given columns
        """
        if isinstance(columns, str):
            columns=[columns]
        bad_cols = [col for col in columns \
                    if col in self.no_diff_cols]
        if bad_cols:
            raise ValueError("Filter column same for all common rows",
                             bad_cols)
        bad_cols = [col for col in columns \
                    if col not in self.diff_cols]
        if bad_cols:
            raise ValueError("Filter column not in both datasets",
                             bad_cols)

        # filter row adds/removes
        df = self.df
        if not row_adds:
            df = df[df._merge != 'right_only']
        if not row_removes:
            df = df[df._merge != 'left_only']
            # filter by column changes
        for col in columns:
            diff_i = df[col +'_1'] != df[col+'_2']
            df = df[diff_i]
        
        return df

_COLUMN_TYPES = []

def _is_col_type(kls):
    """Decorator that adds column type classes to the _COLUMN_TYPES
    list, which is what the col_type() function checks against
    """
    _COLUMN_TYPES.append(kls)
    return kls

@_is_col_type
class IdColumn(object):
    """Values aren't null or duplicated
    """
    def __init__(self, col):
        self.label='id'
        self.applies = False
        self.col = col
        self._disqual = []

        if col.isnull().any():
            self._disqual.append('Null values')

        if col.duplicated().any():
            self._disqual.append('Duplicates')

        if not self._disqual:
            self.applies = True

@_is_col_type
class CategoricalColumn(object):
    """Values are duplicated more than the specified percentage
    """
    def __init__(self, col, cat_dup_min=.5):
        self.label='categorical'
        self.applies = False
        self.col = col
        self._disqual = []

        dup_rate = col.fillna('magic_na').duplicated().mean()
        if dup_rate < cat_dup_min:
            self._disqual.append(
                f"Dup rate {dup_rate} below min {cat_dup_min}")

        if not self._disqual:
            self.applies = True

@_is_col_type
class DateRegularColumn(object):
    """Dates are evenly spaced by number of days or months.

    This automatically catches weeks & years (except for leap years)
    """
    label='date regularly spaced'
    
    def __init__(self, col):
        self.applies = False
        self.col = col
        self._disqual = []

        vals = col.dropna().unique()
        vals = pd.Series(vals).sort_values()

        # check for year
        has_year = vals.apply(lambda x: hasattr(x, 'year'))
        if not has_year.all():
            self._disqual.append("Not all values have .year")
            return
        
        # check for month
        has_month = vals.apply(lambda x: hasattr(x, 'month'))
        if not has_month.all():
            self._disqual.append("Not all values have .month")
            return
        
        # single value
        if len(vals) == 1:
            self._disqual.append("Only one value")
            return

        # Check for same number of months between, and they aren't all
        # in the same month
        month_diffs = set([(post.year-pre.year)*12 +
                           (post.month-pre.month)
                           for post,pre
                           in zip(vals[1:], vals[:-1])])
        if len(month_diffs) > 1:
            self._disqual.append(
                f"Multiple month intervals ({len(month_diffs)})")
        elif month_diffs == set([0]):
            self._disqual.append(
                "All in same month, not enough info")
        else:
            self.applies = True
            return
            
        # Same number of days between unique values
        day_diffs = set([post - pre for post,pre \
                         in zip(vals[1:], vals[:-1])])
        if len(day_diffs) == 1:
            self.applies = True
            self._disqual = []
        else:
            self._disqual.append(
                f"Multiple day intervals ({len(day_diffs)})")

@_is_col_type
class FlagColumn(object):
    """Only two values and high rate of duplication

    NULLS are ignored
    """
    label='flag'
    
    def __init__(self, col, min_dup_rate=.8):
        self.applies = False
        self.col = col
        self._disqual = []

        vals = col.dropna().unique()
        n = len(vals)
        if n != 2:
            self._disqual.append(f"Not two values (n={n})")
            return

        dup_rate = col.dropna().duplicated().mean()
        if dup_rate < min_dup_rate:
            self._disqual.append(
                f"Dup rate {dup_rate:.0%} below threshold "+\
                f"{min_dup_rate:.0%}")
            return

        # otherwise applies
        self.applies = True
        
@_is_col_type
class FlagNullColumn(object):
    """Only one value, rarely populated, rest are nulls
    """
    label='flag null'
    
    def __init__(self, col, min_null_rate=.8):
        self.applies = False
        self.col = col
        self._disqual = []

        null_rate = col.isnull().mean()
        if null_rate < min_null_rate:
            self._disqual.append(
                f"Null rate {null_rate:.0%} below threshold "+\
                f"{min_null_rate:.0%}")
            return
        
        vals = col.dropna().unique()
        n = len(vals)
        if n != 1:
            self._disqual.append(f"More than one value (n={n})")
            return

        # otherwise applies
        self.applies = True
        
@_is_col_type
class NumericNormalColumn(object):
    """Test for normality
    """
    label='num normal'
    
    def __init__(self, col, max_pvalue=.05):
        self.applies = False
        self.col = col
        self._disqual = []

        import scipy.stats
        import warnings
        # scipy/stats/stats.py:1535: UserWarning: kurtosistest only
        # valid for n>=20 ... continuing anyway, n=10
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = scipy.stats.normaltest(col, nan_policy='omit')
        if res.pvalue >= max_pvalue:
            self._disqual.append(
                f"Failed test, p {res.pvalue:.2f} > "+\
                f"{max_pvalue}")
            return
        else:
            self.applies=True
        
@_is_col_type
class NumericLongTailColumn(object):
    """Test for a long tail

    If the KS test says its not dissimilar from any of these
    distros, then call it long tail:
     - lognorm
     - chi2
     - expon
    
    """
    label='num long tail'
    
    def __init__(self, col, max_pvalue=.05):
        self.applies = False
        self.col = col
        self._disqual = []

        from scipy import stats
        import warnings
        # scipy/stats/stats.py:1535: UserWarning: kurtosistest only
        # valid for n>=20 ... continuing anyway, n=10
        for distro in ['lognorm', 'chi2', 'expon']:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = stats.kstest(
                    col, distro,
                    getattr(stats, distro).fit(col))
            if res.pvalue > max_pvalue:
                self._disqual = []
                self.applies = True
                return
            else:
                self._disqual.append(
                    f"Not {distro}, p={res.pvalue:.2f}")
                    
@_is_col_type
class NumericAccountingColumn(object):
    """Test for a values at all different scales of 10x

    Applies if there are at least three different scales of 10x, such
    as:
            0
          112
       50,000

    Ignores nulls
    """
    label='num accounting'
    
    def __init__(self, col, max_pvalue=.05):
        self.applies = False
        self.col = col
        self._disqual = []

        vals = col.dropna().unique()
        import math
        f = lambda x: int(math.log10(x)) if x != 0 else 0
        vals = set([f(_) for _ in vals])
        self._disqual = f'len {len(vals)}'
        if len(vals) >= 3:
            self.applies = True
        else:
            self._disqual = 'Only {} levels of 10x ({})'.format(
                len(vals), vals)

            
@_is_col_type
class TextColumn(object):
    """Criteria:
      - strings
      - few repeated values (not categorical)
      - strings are more than 20 characters
    """
    label='text'
    
    def __init__(self, col, str_len_min=20, cat_dup_max=.5):

        self._disqual = []

        # check for str attribute
        if not hasattr(col, 'str'):
            self._disqual.append("Doesn't have str attribute")
            return

        # copy and remove nulls
        col = col.copy().dropna()
        
        # check only strings
        col_classes= [_ for _ in
                      col.apply(lambda x: type(x).__name__).unique()]
        if col_classes != ['str']:
            self._disqual.append(
                f"Non-string class: {col_classes}")

        # check duplication rate
        dup_rate = col.duplicated().mean()
        if dup_rate > cat_dup_max:
            self._disqual.append(
                f"Dup rate {dup_date} above max {cat_dup_max}")

        # check string length
        mean_str_len = col.str.len().mean()
        if mean_str_len < str_len_min:
            self._disqual.append(
                f"Mean string length {mean_str_len} "+\
                f"below {str_len_min}")

    @property
    def applies(self):
        return not self._disqual 
            
def col_type(col, types=_COLUMN_TYPES):
    """Return all applicable column types for the provided column

    Returns a tuple of (applies, disqualified), where each
    are a dictionary with keys of column type labels and values of
    column type instances.
    """
    applies = {}
    disqual = {}
    for ct_class in _COLUMN_TYPES:
        try:
            ct = ct_class(col)
        except Exception as e:
            x = lambda: None
            x._disqual = e.args
            disqual[ct_class.label] = x
            continue
        if ct.applies:
            applies[ct.label] = ct
        else:
            disqual[ct.label] = ct
    return (applies, disqual)
        
class ColTypeDf(object):
    """Determine column type for all columns in a dataframe and
    provide text summary, one column per line
    """
    def __init__(self, df):
        self._col_types = {k: col_type(col) for k,col in df.iteritems()}
        
    @property
    def summary(self):
        """For each field, list col types that apply
        """
        s = []
        for col_name, (applies, disqual) in self._col_types.items():
            s.append("{:20s}: {}".format(col_name, ", ".join(applies)))
        return "\n".join(s)
            
    def pprint(self):
        print(self.summary)
