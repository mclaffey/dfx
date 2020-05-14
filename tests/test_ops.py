#!/usr/bin/env python3
import sys
import datetime

import pandas as pd
import numpy as np

import dfx.ops as ops
import unittest_dfx

class ReduceTest(unittest_dfx.AbstractDfTestCase):
    """Test the identification and removal of columns with single values
    """

    def setUp(self):
        self.df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'all_null': [None] * 5,
            'all_nan': [np.NaN] * 5,
            'all_zero': [0] * 5,
            'all_hello': ['hello'] * 5
            })
        self.rdf = ops.ReducedDf(self.df)

    def test_equals(self):
        """Test the equality funcitonality with equal instances
        """
        rdf2 = ops.ReducedDf(self.df)
        self.assertEqual(self.rdf, rdf2)

    def test_not_equal(self):
        """Test the equality funcitonality with dissimilar instances
        """
        rdf2 = ops.ReducedDf(self.df[3:])
        self.assertNotEqual(self.rdf, rdf2)
        
    def test_nulls(self):
        """Detect columns that are null for all rows"""
        self.assertEqual(self.rdf.nulls, ['all_null', 'all_nan'])

    def test_zeros(self):
        """Detect columns that are zero for all rows"""
        self.assertEqual(self.rdf.zeros, ['all_zero'])

    def test_constants(self):
        """Detect columns that have a single value for all rows
        """
        self.assertEqual(self.rdf.constants, {'all_hello': 'hello'})

    def test_varied(self):
        """Detect columns that have more than one value
        """
        self.assertEqualDf(self.rdf.df, self.df[['id']])

    def test_summary(self):
        """Make sure the summary contains some expected values

        """
        s = self.rdf.summary
        # summary lines
        self.assertIn('nulls', s.lower())
        self.assertIn('zero', s.lower())
        self.assertIn('constants', s.lower())
        # column names all there
        for col in self.df.columns:
            self.assertIn(col, s)
        # some row values
        self.assertIn('5', s)
        self.assertIn('hello', s)

    def test_best(self):
        """Find a way to divide the dataset rows to provide a more
        efficient summary
        """
        df = pd.DataFrame({
            'id': [1,2,3,4,5],
            'region': ['east'] * 3 + ['west'] * 2,
            'manager': ['acriminos hult'] * 3 + ['jack black'] * 2,
            'state': ['New York'] * 3 + ['California'] * 2,
            'department': ['blindess widget production'] * 3 + \
            ['strategy and innovation'] * 2             
            })
        brdf = ops.BestReducedDf(df)
        self.assertIn('region', brdf.split_columns)

        east_rdf = ops.ReducedDf(df.query("region == 'east'"))
        west_rdf = ops.ReducedDf(df.query("region == 'west'"))

        self.assertEqual(brdf.east, east_rdf)
        self.assertEqual(brdf.west, west_rdf)
        
        # just make sure this runs without raise an exception,
        # not checking content
        s = brdf.summary

class DiffTest(unittest_dfx.AbstractDfTestCase):
    """Test the differences between to datasets
    """
    
    def test_row_differences(self):
        """Detect rows that have been added and removed
        """
        df1 = pd.DataFrame({
            'val': [10, 20, 30],
            },
                           index=[1,2,3])
        df2 = pd.DataFrame({
            'val': [10, 40, 50],
            },
                           index=[1,4,5])
        diff = ops.DiffDf(df1, df2)

        self.assertEqualDf(diff.rows_added, df2.loc[[4,5]])
        self.assertEqualDf(diff.rows_removed, df1.loc[[2,3]])

    def test_col_counts(self):
        """Count the number of values that have changed by column
        """
        df1 = pd.DataFrame({
            'val_a': [10, 20, 30],
            'val_b': [100, 200, 300],
            'val_c': [0, 0, 0],
            },
                           index=[1,2,3])
        df2 = pd.DataFrame({
            'val_a': [10, 21, 31],
            'val_b': [100, 200, None],
            'val_c': [0, 0, 0],
        },
                           index=[1,2,3])
        diff = ops.DiffDf(df1, df2)
        exp_col_counts = pd.Series(
            [2, 1, 0],
            index=['val_a', 'val_b', 'val_c'],
            name='diff_counts'
            )
        self.assertEqualSeries(diff.col_diff_counts, exp_col_counts)

    def test_side_by_side(self):
        """Provide side-by-side before and after values for rows
        that existed in both datases
        """
        df1 = pd.DataFrame({
            'val_a': [10, 20, 30],
            'val_b': [100, 200, 300],
            'fruit': ['apple', 'apple', 'orange'],
        },
                           index=[1,2,3])
        df2 = pd.DataFrame({
            'val_a': [10, 21, None],
            'val_b': [100, 200, 300],
            'veg': ['beans', 'beans', 'lettuce'],
        },
                           index=[1,2,3])
        diff = ops.DiffDf(df1, df2)
        _merge = pd.Categorical(
            values=['both'] * 3,
            categories=['left_only', 'right_only', 'both'])
        exp_side_by_side_df = pd.DataFrame({
            '_merge' : _merge,
            'val_a_1': [10, 20, 30],
            'val_a_2': [10, 21, None],
            'val_b'  : [100, 200, 300],
            'fruit_1': ['apple', 'apple', 'orange'],
            'veg_2'  : ['beans', 'beans', 'lettuce'],
            },
                                           index=[1,2,3])
        act_diff_df = diff.df
        # 
        self.assertEqualDf(diff.df, exp_side_by_side_df)

    def test_filtered_changes(self):
        """Filter to rows that had a value change in a given column
        """
        df1 = pd.DataFrame({
            'val_a': [10, 20, 30],
            'val_b': [100, 200, 300],
            'fruit': ['apple', 'apple', 'orange'],
        },
                           index=[1,2,3])
        df2 = pd.DataFrame({
            'val_a': [10, 21, None],
            'val_b': [100, 200, 300],
            'veg': ['beans', 'beans', 'lettuce'],
        },
                           index=[1,2,3])
        diff = ops.DiffDf(df1, df2)

        val_a = diff.changed('val_a')
        self.assertEqual(val_a.shape[0], 2)
        
        with self.assertRaises(ValueError):
            val_a = diff.changed('val_b')
        with self.assertRaises(ValueError):
            val_a = diff.changed('fruit')
        with self.assertRaises(ValueError):
            val_a = diff.changed('veg')
        
class ColTypeTest(unittest_dfx.AbstractDfTestCase):
    """Test the automatic identification of column types based on
    values
    """

    def assertOnlyColType(self, col_type_label, applies, disquals):
        self.assertHasColType(col_type_label, applies, disquals,
                              just_one=True)

    def assertNotColType(self, col_type_label, applies, disquals):
        # Fail when expected column type applies
        if col_type_label in applies:
            self.fail(f"Col_type '{col_type_label}' applies")
        # Fail when expected column type not found in disqual
        if not col_type_label in disquals:
            self.fail(f"Col_type '{col_type_label}' not in "+\
                      f"disquals: {disquals}")
        # otherwise pass

    def assertHasColType(self, col_type_label, applies, disquals,
                         just_one=False):        
        # When expected column type applies
        if col_type_label in applies:
            other_col_types = [_ for _ in applies
                               if _ != col_type_label]
            if not other_col_types or not just_one:
                return # pass
            else:
                self.fail(("In addition to '{}', other unexpected "+\
                           "col types: {}").format(
                               col_type_label
                               ,other_col_types))
                
        # When expected column type was disqualified
        if col_type_label in disquals:
            ct = disquals[col_type_label]
            if not hasattr(ct, '_disqual'):
                self.fail(
                    f"Found expected column type "+\
                    f"'{col_type_label}' as disqualified, "+\
                    f"but no _disqual attribute for reason. {ct}")
            elif not ct._disqual:
                self.fail(
                    f"Found expected column type "+\
                    f"'{col_type_label}' as disqualified, "+\
                    f"but no reason. {ct}")
            else:
                self.fail(
                    f"Found expected column type "+\
                    f"'{col_type_label}' as disqualified, "+\
                    f"reasons: {ct._disqual}")
                
        # When expected column type wasn't found
        applies_str = 'None'
        if applies:
            applies_str = ", ".join(applies.keys())
        disquals_str = 'None'
        if disquals:
            disquals_str = ", ".join(disquals.keys())
            except_str = "\n--" + "\n--".join(
                [f"{k}: {v.args[0]}" for k,v in disquals.items() \
                 if isinstance(v, Exception)]) + '\n'
            
        self.fail((f"{except_str}"+\
                   "Expected column type '{}' not found.\n"+\
                   "Applies: {}.\nDisquals: {}").format(
                       col_type_label
                       ,applies_str
                       ,disquals_str))
                          
    def test_id(self):
        """Detect an ID column
        """
        col = pd.Series([1,2,3,4,5])
        (cts, disquals) = ops.col_type(col)
        self.assertHasColType('id', cts, disquals)

    def test_categorical_alpha(self):
        """Detect a categorical field with string values
        """
        col = pd.Series(['cat'] * 3 + ['dog'])
        (cts, disquals) = ops.col_type(col)
        self.assertOnlyColType('categorical', cts, disquals)

    def test_categorical_num(self):
        """Detect a categorical field with numerical values
        """
        col = pd.Series([10] * 3 + [20])
        (cts, disquals) = ops.col_type(col)
        self.assertHasColType('categorical', cts, disquals)

    def test_date_reg(self):
        """Detect a date field with regularly spaced values
        """
        col = pd.Series([datetime.date(2020, i, 1) \
                         for i in range(1, 7)])
        (cts, disquals) = ops.col_type(col)
        self.assertHasColType(ops.DateRegularColumn.label,
                              cts, disquals)        

    def test_date_irreg(self):
        """Detect a date field with irregularly spaced values
        """
        col = pd.Series([
            datetime.date(2020, 1, 15),
            datetime.date(2020, 1, 27),
            datetime.date(2020, 2, 1),
            datetime.date(2020, 9, 30),
            datetime.date(2020, 10, 31),
            ])        
        (cts, disquals) = ops.col_type(col)
        self.assertNotColType(ops.DateRegularColumn.label,
                              cts, disquals)
        
    def test_num_accounting(self):
        """Detect a numeric value that looks like accounting totals
        """
        col = pd.Series([0,0,0,100,110, 1000, 1_000_000,
                         50_000, 0, 7000])
        (cts, disquals) = ops.col_type(col)
        self.assertHasColType('num accounting', cts, disquals)
                        
    def test_num_normal(self):
        """Detect an normally distributed numeric column
        """
        col = pd.Series([1,2,3,3,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,7,7,8,11])
        (cts, disquals) = ops.col_type(col)
        self.assertHasColType('num normal', cts, disquals)
        
    def test_num_long_tail(self):
        """Detect an numeric column with long tail distribution
        """
        col = pd.Series([1,1,1,1,1,2,2,2,2,3,3,3,4,4,5,5,6,6,8,10])
        (cts, disquals) = ops.col_type(col)
        self.assertHasColType('num long tail', cts, disquals)        
        
    def test_flag(self):
        """Detect a column with two values, a common and a rare
        """
        col = pd.Series(['Y'] * 17 + ['N'] * 3)
        (cts, disquals) = ops.col_type(col)
        self.assertHasColType('flag', cts, disquals)
        
    def test_flag_null(self):
        """Detect an column that is mostly null except for a small
        percent of rows
        """
        col = pd.Series([None] * 18 + [1] * 2)
        (cts, disquals) = ops.col_type(col)
        self.assertHasColType('flag null', cts, disquals)        
        
    def test_text(self):
        """Detect a column with free form text
        """
        col = pd.Series([
            'this is a comment',
            None,
            'this is a comment',
            'this is a much longer comment that contains more words',
            ''])
        cts = ops.col_type(col)
        self.assertColType(cts, ['text'])

    def assertColType(self, cts, expected_cts_labels):
        """Test if the actual column type labels match
        expected
        """
        applies, disquals = cts
        act_cts_labels = applies.keys()
        act_cts_labels = sorted(act_cts_labels)
        expected_cts_labels = sorted(expected_cts_labels)
        for label in expected_cts_labels:
            if label in applies:
                pass
            elif label not in disquals:
                self.fail(f"Col type label '{label}' not in "+\
                          "applies or disqual")
            else:
                print(f"Not col type '{label}' because: " +\
                      str(disquals[label]._disqual))
        self.assertEqual(act_cts_labels, expected_cts_labels)
        
if __name__ == '__main__':
    sys.exit(unittest_dfx.main(__file__))
