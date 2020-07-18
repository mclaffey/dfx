#!/usr/bin/env python
import os
import sys
import unittest

import unittest_dfx

import dfx.ui_curses
import dfx.datasets

class EnvTest(unittest_dfx.AbstractDfTestCase):
    """Test functionality of ui_curses.Environment
    """

    def test_load_file(self):
        """Providing a path to a csv loads a new Dfview named
        after the path
        """
        env = dfx.ui_curses.Environment()
        data_path='../sample_data/obesity.csv'
        data_dir=os.path.dirname(data_path)
        env.load_file(data_path)

        self.assertIn(data_path, env.dfvs)
        self.assertEqual(data_path, env.current_dfv_name)
        self.assertEqual(data_dir, env.current_dir)
        self.assertTrue(isinstance(env.dfv, dfx.ui_curses.DfView))
                        
    def test_next_file(self):
        """Specify a directory and calling next file loads new csvs
        """
        env = dfx.ui_curses.Environment()
        env.current_dir='../sample_data'

        file_name = env.next_file()
        self.assertEqual(file_name, '../sample_data/corona.csv')
        self.assertEqual(env.current_dfv_name,
                         '../sample_data/corona.csv')

        file_name = env.next_file()
        self.assertEqual(file_name, '../sample_data/emissions.csv')
        self.assertEqual(env.current_dfv_name,
                         '../sample_data/emissions.csv')
        
    def test_new_dfv(self):
        """Test convenience method for creating a new DfView
        """
        env=dfx.ui_curses.Environment()
        df=dfx.datasets.checks
        dfv_p=dfx.ui_curses.DfView(df)
        env.new_dfv(df=df, name='child', parent_dfv=dfv_p)

        self.assertIn('child', env.dfvs)
        self.assertEqual('child', env.current_dfv_name)
        self.assertTrue(isinstance(env.dfv, dfx.ui_curses.DfView))
        self.assertEqual(env.dfv.parent_dfv, dfv_p)

    def test_next(self):
        """Next goes to alphabetically next Dfview
        """
        env=dfx.ui_curses.Environment()
        df=dfx.datasets.checks
        env.new_dfv(df=df, name='abc')
        env.new_dfv(df=df, name='xyz')        

        self.assertEqual(env.current_dfv_name, 'xyz')
        env.next()
        self.assertEqual(env.current_dfv_name, 'abc')
        env.next()
        self.assertEqual(env.current_dfv_name, 'xyz')        

if __name__=='__main__':
    sys.exit(unittest_dfx.main(__file__))
