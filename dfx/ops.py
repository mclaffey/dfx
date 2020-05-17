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

# ##################################################################
#
# value patterns

_VALUE_PATTERN_CLASSES = []

def _is_value_pattern(kls):
    """Decorator that adds value pattern classes to the _VALUE_PATTERN_CLASSES
    list, which is what the value_patterns() function checks against
    """
    _VALUE_PATTERN_CLASSES.append(kls)
    return kls

class AbstractValuePattern(object):
    """Base class for common value pattern properties
    """
    def __init__(self, col):
        self.col = col
        self.disqual = []
        self.detail = ''
        self.evaluate()

    @property
    def applies(self):
        return len(self.disqual)==0

    def evalute(self):
        """To be implemented by each subclass.

        Evaluates if the column meets the value pattern. If not, this
        method must add reasons (plain strings) to self.disqual.

        When this method returns to __init__(), if self.disqual is
        still empty, the pattern is assumed to apply.
        """
        raise NotImplementedError('Must be implemented by subclass')

    def __str__(self):
        return "{}, {:20s}, {}{}{}".format(
            self.col.name,
            self.label,
            'applies' if self.applies else 'disqualified',
            '='+",".join(self.disqual) if self.disqual else '',
            ', ' + self.detail if self.detail or self.disqual else 'no detail'
            )

# do not apply @_is_value_pattern to this
class DummyValuePattern(AbstractValuePattern):
    """An instance of ValuePattern that stores the required attributes
    but doesn't have any rule implementation
    """
    label='dummy'
    def evaluate(self):
        pass
        
@_is_value_pattern
class IdPattern(AbstractValuePattern):
    """Values aren't null or duplicated
    """
    label='id'
    def evaluate(self):
        null_count = self.col.isnull().sum()
        if null_count:
            self.disqual.append('{} null values'.format(null_count))            

        dup_count = self.col.duplicated().sum()
        if dup_count:
            self.disqual.append('{} duplicates'.format(dup_count))

        if not self.disqual:
            self.detail='All unique, no nulls'

@_is_value_pattern
class CategoricalPattern(AbstractValuePattern):
    """Values are duplicated more than the specified percentage
    """
    label='categorical'
    cat_dup_min=.5

    
    def evaluate(self):
        cat_dup_min = self.cat_dup_min
        dup_rate = self.col.fillna('magic_na').duplicated().mean()

        if dup_rate < cat_dup_min:
            self.disqual.append(
                "Dup rate {:.0%} below minimum of {:.0%}".format(
                    dup_rate, cat_dup_min))
        else:
            self.detail=(
                "Applies because dup rate {:.0%} is more "
                "than threshold of {:.0%}".format(
                    dup_rate, cat_dup_min))            
            

@_is_value_pattern
class DateRegularPattern(AbstractValuePattern):
    """Dates are evenly spaced by number of days or months.

    This automatically catches weeks & years (except for leap years)
    """
    label='date regularly spaced'
    
    def evaluate(self):

        vals = self.col.dropna().unique()
        vals = pd.Series(vals).sort_values()

        # check for year
        has_year = vals.apply(lambda x: hasattr(x, 'year'))
        if not has_year.all():
            self.disqual.append("Not all values have .year")
            return
        
        # check for month
        has_month = vals.apply(lambda x: hasattr(x, 'month'))
        if not has_month.all():
            self.disqual.append("Not all values have .month")
            return
        
        # single value
        if len(vals) == 1:
            self.disqual.append("Only one value")
            return

        # Check for same number of months between, and they aren't all
        # in the same month
        month_diffs = set([(post.year-pre.year)*12 +
                           (post.month-pre.month)
                           for post,pre
                           in zip(vals[1:], vals[:-1])])
        if len(month_diffs) > 1:
            self.disqual.append(
                f"Multiple month intervals ({len(month_diffs)})")
        elif month_diffs == set([0]):
            self.disqual.append(
                "All in same month, not enough info")
        else:
            return
            
        # Same number of days between unique values
        day_diffs = set([post - pre for post,pre \
                         in zip(vals[1:], vals[:-1])])
        if len(day_diffs) == 1:
            self.disqual = []
        else:
            self.disqual.append(
                f"Multiple day intervals ({len(day_diffs)})")

@_is_value_pattern
class FlagPattern(AbstractValuePattern):
    """Only two values and high rate of duplication

    NULLS are ignored
    """
    label='flag'
    min_dup_rate=.8

    def evaluate(self):

        vals = self.col.dropna().unique()
        n = len(vals)
        if n != 2:
            self.disqual.append(f"Not two values (n={n})")
            return

        dup_rate = self.col.dropna().duplicated().mean()
        if dup_rate < self.min_dup_rate:
            self.disqual.append(
                f"Dup rate {dup_rate:.0%} below threshold "+\
                f"{min_dup_rate:.0%}")
            return

        
@_is_value_pattern
class FlagNullPattern(AbstractValuePattern):
    """Only one value, rarely populated, rest are nulls
    """
    label='flag null'
    min_null_rate=.8

    def evaluate(self):
        null_rate = self.col.isnull().mean()
        if null_rate < self.min_null_rate:
            self.disqual.append(
                ("Null rate {:.0%} below threshold "
                 "{:.0%}").format(
                     null_rate,
                     self.min_null_rate))
            return
        
        vals = self.col.dropna().unique()
        n = len(vals)
        if n != 1:
            self.disqual.append(f"More than one value (n={n})")
            return
        
@_is_value_pattern
class NumericNormalPattern(AbstractValuePattern):
    """Test for normality
    """
    label='normal'
    max_pvalue=.05
    
    def evaluate(self):
        import scipy.stats
        import warnings
        # scipy/stats/stats.py:1535: UserWarning: kurtosistest only
        # valid for n>=20 ... continuing anyway, n=10
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = scipy.stats.normaltest(self.col, nan_policy='omit')
        self.details = str(res)
        if res.pvalue < self.max_pvalue:
            self.disqual.append(
                ("Test indicates not normal, p {:.2f} < "
                 "{:.2f}").format(
                     res.pvalue,
                     self.max_pvalue))
        
@_is_value_pattern
class NumericLongTailPattern(AbstractValuePattern):
    """Test for a long tail

    If the KS test says its not dissimilar from any of these
    distros, then call it long tail:
     - lognorm
     - chi2
     - expon
    
    """
    label='num long tail'
    max_pvalue=.05

    def evaluate(self):
        from scipy import stats
        import warnings
        # scipy/stats/stats.py:1535: UserWarning: kurtosistest only
        # valid for n>=20 ... continuing anyway, n=10
        for distro in ['lognorm', 'chi2', 'expon']:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = stats.kstest(
                    self.col, distro,
                    getattr(stats, distro).fit(self.col))
            if res.pvalue > self.max_pvalue:
                # if any test fail to reject null hypothesis, clear
                # out any disqualifications from any previous tests
                # and return (forcing applies=True)
                self.disqual = []
                return
            else:
                self.disqual.append(
                    f"Not {distro}, p={res.pvalue:.2f}")
                    
@_is_value_pattern
class NumericMultiscalePattern(AbstractValuePattern):
    """Test for a values at all different scales of 10x

    Applies if there are at least three different scales of 10x, such
    as:
            0
          112
       50,000

    Ignores nulls
    """
    label='multi_scale'
    min_scale_count = 3
    
    def evaluate(self):
        vals = self.col.dropna().unique()
        import math
        # determine number of digits
        f = lambda x: int(math.log10(x)) if x != 0 else 0
        digit_counts = set([f(_) for _ in vals])
        self.details = 'Log10 scales that exist: {}'.format(
            digit_counts)
        if len(digit_counts) < self.min_scale_count:
            self.disqual = 'Only {} levels of 10x ({})'.format(
                len(digit_counts), digit_counts)

            
@_is_value_pattern
class TextPattern(AbstractValuePattern):
    """Criteria:
      - strings
      - few repeated values (not categorical)
      - strings are more than 20 characters
    """
    label='text'
    str_len_min=20
    cat_dup_max=.5

    def evaluate(self):
        col = self.col

        # check for str attribute
        if not hasattr(col, 'str'):
            self.disqual.append("Doesn't have str attribute")
            return

        # copy and remove nulls
        col = col.copy().dropna()
        
        # check only strings
        col_classes= [_ for _ in
                      col.apply(lambda x: type(x).__name__).unique()]
        if col_classes != ['str']:
            self.disqual.append(
                "Non-string class: {}".format(col_classes))

        # check duplication rate
        dup_rate = col.duplicated().mean()
        if dup_rate > self.cat_dup_max:
            self.disqual.append(
                "Dup rate {:.0%} above max {:.0%}".format(
                    dup_rate,
                    self.cat_dup_max
                    ))

        # check string length
        mean_str_len = col.str.len().mean()
        if mean_str_len < self.str_len_min:
            self.disqual.append(
                "Mean string length {} below {}".format(                    
                    mean_str_len,
                    self.str_len_min
                    ))

def value_patterns(col, pattern_classes=_VALUE_PATTERN_CLASSES):
    """Return all applicable value patterns for the column

    Returns a tuple of (applies, disqualified), where each
    are a dictionary with keys of the value pattern label
    and values of the value pattern instance.
    """
    applies = {}
    disqual = {}
    for vp_class in pattern_classes:
        try:
            vp = vp_class(col)
        except Exception as e:
            #raise e from None
            vp=DummyValuePattern(col)
            vp.label   = vp_class.label
            vp.disqual = e.args
        if vp.applies:
            applies[vp.label] = vp
        else:
            disqual[vp.label] = vp
    return (applies, disqual)
        
class ValuePatternsDf(object):
    """Determine value pattern for all columns in a dataframe and
    provide text summary, one column per line

    This is a convenience class for calling value_pattern()
    and displaying the results.    
    """
    def __init__(self, df, display=False):
        """Get value pattern for each column in dataframe

        :df - pandas.DataFrame
        :display - optional, default False. If true, prints
           the value patterns of each column as it calculates
           them        
        """
        self.df = df
        self.value_patterns = {}
        if display:
            print("{:20.20s}  {:30.30s} {:20.20s} {:20.20s}".format(
                'Column',
                'Value pattern',
                'First value',
                'Last value',
            ))
            print("{:20.20s}  {:30.30s} {:20.20s} {:20.20s}".format(
                '------',
                '-------------',
                '-----------',
                '----------',
            ))
        for col_name, col in df.iteritems():
            self.value_patterns[col_name] = value_patterns(col)
            if display:
                print("{:20.20s}: {:30.30s} {:20.20s} {:20.20s}".format(
                    col_name,
                    ", ".join(self.value_patterns[col_name][0]),
                    str(col.values[0]),
                    str(col.values[-1]),
                ))
        
    def __str__(self):
        """For each field, list value patterns that apply
        """
        s = []
        for col_name, (applies, disqual) in self.value_patterns.items():
            s.append("{:20s}: {}".format(col_name, ", ".join(applies)))
        return "\n".join(s)
