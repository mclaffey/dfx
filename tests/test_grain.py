#!/usr/bin/env python
import sys
import unittest

import unittest_dfx
import dfx.datasets as datasets
import dfx.grain as dfx

class TestGrainFind(unittest_dfx.AbstractDfTestCase):
    """Test methods for identifying the columns in a dataset
    that uniquely identify records
    """

    def test_single_col(self):
        """Find a single column that uniquely identifies
        a dataset
        """
        df = datasets.employees
        g = dfx.GrainDf(df)
        self.assertEqual(True, g.unique)
        self.assertEqual(g.columns, ('employee_id', ))

    def test_two_cols(self):
        """Find two columns that uniquely identify rows in a dataset
        """
        df = datasets.employee_hist
        g = dfx.GrainDf(df)
        self.assertEqual(True, g.unique)
        self.assertEqual(g.columns, ('employee_id', 'status'))

    def test_multi_cols(self):
        """Find multiple columns that uniquely identify rows in a
        dataset
        """
        df = datasets.cube
        g = dfx.GrainDf(df)
        self.assertEqual(True, g.unique)
        self.assertEqual(g.columns, ('dept', 'status', 'date'))

    def test_not_unique_single_col(self):
        """Recognize when a column almost uniquely identifies a
        dataset but there are duplicates
        """
        df = datasets.employee_dups
        g = dfx.GrainDf(df, uniq_threshold=.6)
        self.assertEqual(False, g.unique)
        self.assertEqual(g.columns, ('employee_id',))

    def test_dup_values_single_col(self):
        """Return the values that are causing duplicates
        """
        df = datasets.employee_dups
        g = dfx.GrainDf(df, uniq_threshold=.6)
        ids = g.duplicate_ids().iloc[:,1].to_list()
        self.assertEqual(ids, ['12345', '24543'])

    def test_dup_values_multi_cols(self):
        """Return the value combinations that are causing duplicates
        in a multi-col grain
        """
        df = datasets.cube
        g = dfx.GrainDf(df, columns=['dept', 'status'],
                        uniq_threshold=0)
        exp_df = df.groupby(['dept', 'status']).size()\
                 .reset_index().rename(columns={0: 'row_count'})\
                 .iloc[:, [2,0,1]]
        self.assertEqualDf(g.duplicate_ids(), exp_df)

    def test_dup_rows_single_col(self):
        """Return the full rows that have duplicated values in a
        single ID column
        """
        df = datasets.employee_dups
        g = dfx.GrainDf(df, uniq_threshold=.6)
        exp_dups = df[df.employee_id.isin(['12345', '24543'])]\
                   .reset_index(drop=True)\
                   .sort_values(['employee_id'])
        act_dups = g.duplicate_rows().drop('row_count', axis=1)
        self.assertEqualDf(act_dups, exp_dups)

    def test_dup_rows_multi_cols(self):
        """Return the full rows that have duplicated value
        combinations in a multi-col grain dataset
        """
        df = datasets.cube
        g = dfx.GrainDf(df, columns=['dept', 'status'],
                        uniq_threshold=0)
        exp_dups = df.sort_values(['dept', 'status', 'date'])
        act_dups = g.duplicate_rows().drop('row_count', axis=1)
        self.assertEqualDf(act_dups, exp_dups)

class TestGrainSummarize(unittest.TestCase):
    """Test the information that is reported about ID columns
    """

    def test_contrib_single_col(self):
        """A single column ID field should always be 100%
        """
        df = datasets.employees
        g = dfx.GrainDf(df)
        self.assertEqual(g.contrib['employee_id'], 1)

    def test_contrib_sparse(self):
        """A two-col sparse grain has concentrated contrib in one
        column
        """
        df = datasets.employee_hist
        g = dfx.GrainDf(df)
        self.assertAlmostEqual(g.contrib['employee_id'], .8, 1)        
        self.assertAlmostEqual(g.contrib['status'], .2, 1)

    def test_contrib_multi_col_hier(self):
        """Multi-col hier grain has most of the contrib at the bottom
        """
        df = datasets.regions
        g = dfx.GrainDf(df)
        self.assertAlmostEqual(g.contrib['state'], .4, 1)
        self.assertAlmostEqual(g.contrib['town'], .6, 1)

    def test_contrib_multi_col_cube(self):
        """Multi-col cube grain has distributed contrib
        """
        df = datasets.cube
        g = dfx.GrainDf(df)
        self.assertAlmostEqual(g.contrib['dept'], .3, 1)
        self.assertAlmostEqual(g.contrib['status'], .3, 1)
        self.assertAlmostEqual(g.contrib['date'], .3, 1)

    def test_relationship_1_many(self):
        """Detect a 1:many relationship
        """
        df = datasets.employee_hist
        g = dfx.GrainDf(df)
        rel = g._get_col_rel('manager_id', 'employee_id')
        self.assertEqual(rel, '1:many')
        

    def test_relationship_many_many_hier(self):
        """Detect a many:many relationship in hierarchy
        """
        df = datasets.regions
        g = dfx.GrainDf(df,
                        columns=['state', 'county', 'town'],
                        force=True)
        self.assertEqual(g.col_rels(), [
            ('state', 'county', 'many:many'),
            ('state', 'town', 'many:many'),
            ('county', 'town', 'many:many')])

    def test_relationship_many_many_cube(self):
        """Detect a many:many relationship in cube
        """
        df = datasets.cube
        g = dfx.GrainDf(df)
        self.assertEqual(g.col_rels(), [
            ('dept', 'status', 'many:many'),
            ('dept', 'date', 'many:many'),
            ('status', 'date', 'many:many')])

    def test_perfect_single_col(self):
        """A single-col grain is always perfect
        """
        df = datasets.employees
        g = dfx.GrainDf(df)
        self.assertTrue(g.perfect)

    def test_perfect_sparse(self):
        """A two-col grain with 99% contrib on one column is not
        perfect
        """
        df = datasets.employee_hist
        g = dfx.GrainDf(df)
        self.assertFalse(g.perfect)

    def test_perfect_hier(self):
        """A multi-col hier grain is not perfect
        """
        df = datasets.regions
        g = dfx.GrainDf(df)
        self.assertFalse(g.perfect)

    def test_perfect_cube(self):
        """A multi-col perfect cube is perfect
        """
        df = datasets.cube
        g = dfx.GrainDf(df)
        self.assertTrue(g.perfect)

    def test_missing_single_col(self):
        """A single-col grain doesn't have a concept of
        perfect/missing, so throw exception
        """
        df = datasets.employees
        g = dfx.GrainDf(df)
        self.assertEqual(g.missing_rows.empty, True)
        self.assertEqual(g.missing_rate, 0)

    def test_missing_sparce(self):
        """A two-col grain with 99% on one column has a lot of missing
        records
        """
        df = datasets.employee_hist
        g = dfx.GrainDf(df)
        self.assertEqual(g.missing_rows.empty, False)
        self.assertAlmostEqual(g.missing_rate, .4, 1)

    def test_missing_hier(self):
        """A multi-col hier grain has a lot of missing records
        """
        df = datasets.regions
        g = dfx.GrainDf(df)
        self.assertEqual(g.missing_rows.empty, False)
        self.assertAlmostEqual(g.missing_rate, .7, 1)

    def test_missing_cube(self):
        """A perfect cube has nothing missing
        """
        df = datasets.cube
        g = dfx.GrainDf(df)
        self.assertEqual(g.missing_rows.empty, True)
        self.assertEqual(g.missing_rate, 0)
        
class TestManipulateDf(unittest_dfx.AbstractDfTestCase):
    """Test ability to manipulate datasets based on grain
    """

    def test_filter_up_sparse(self):
        """For IDs that have the sparse values, show all records        
        """
        df = datasets.employee_hist
        g = dfx.GrainDf(df)
        act_rows = g.status.filter_up('old')

        # get expected rows
        emp_ids = df.employee_id[df.status=='old']
        exp_rows = df[df.employee_id.isin(emp_ids)]\
                   .sort_values(['employee_id', 'status'])
        
        self.assertEqualDf(act_rows, exp_rows)

    def test_filter_up_hier(self):
        """For a hierarchy, show all sibling rows
        """
        df = datasets.regions
        g = dfx.GrainDf(df, columns=['state', 'county', 'town'],
                        force=True)
        act_rows = g.town.filter_up('penfield')

        # get expected rows
        rows = df[df.town=='penfield']
        ids = rows.groupby(['state', 'county']).size().index
        indexed_df = df.set_index(['state', 'county'])
        exp_rows = indexed_df.loc[ids].reset_index()
                
        self.assertEqualDf(act_rows, exp_rows)

    def test_filter_up_hier_also_above(self):
        """Filter up but ignore this additional field
        """
        df = datasets.regions
        g = dfx.GrainDf(df, columns=['state', 'county', 'town'],
                        force=True)
        act_rows = g.town.filter_up('penfield', also_above='county')

        # get expected rows
        rows = df[df.town=='penfield']
        ids = rows.groupby(['state']).size().index
        indexed_df = df.set_index(['state', 'county'])
        exp_rows = indexed_df.loc[ids].reset_index()
        
        self.assertEqualDf(act_rows, exp_rows)



        
if __name__ == '__main__':
    sys.exit(unittest_dfx.main(__file__))
