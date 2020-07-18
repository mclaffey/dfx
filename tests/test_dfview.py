#!/usr/bin/env python
import os
import sys
import unittest

import unittest_dfx

sys.path.append('..')
import dfx.ui_curses
import dfx.datasets

class DfViewTest(unittest_dfx.AbstractDfTestCase):
    """Test functionality of ui_curses.DfView

    set_preview
    row_preview
    
    find, find_status
    grain
    melt
    pivot

    sort
    
    col_widths_orig
    
    add_grain_column
    remove_grain_column
    

    
    """

    def test_set_preview(self):
        """Set preview to a specific row, then use offset

        This first selects a row which is towards the beginning, so
        the middle rows are those immediately after head and the row
        selected is at the end of those middle rows.

        Then this offsets +1. That new row is not currently in middle
        rows, so middle rows are reset to to be those immediately
        before tail and the row selected is at the beginning of those
        middle rows.

        """
        df  = dfx.datasets.checks
        dfv = dfx.ui_curses.DfView(df)
        del df['company'] # Reduced constant

        # set to specific row
        dfv.set_preview(center_i=14)
        preview=dfv.row_preview
        self.assertEqual(14, dfv.row_selected)
        self.assertIn('row_number', preview.columns)
        self.assertEqual(15, dfv.row_preview.iloc[
            dfv.row_selected]['row_number'])
        del preview['row_number'] # remove since not in original        
        self.assertEqualDf(preview,
                           df.iloc[[0,1,2,3,4,
                                    5,6,7,8,9,
                                    10,11,12,13,14,
                                    25,26,27,28,29
                                   ]])

        # offset
        dfv.set_preview(offset=1)
        preview=dfv.row_preview
        self.assertEqual(5, dfv.row_selected)
        self.assertIn('row_number', preview.columns)
        self.assertEqual(16, dfv.row_preview.iloc[
            dfv.row_selected]['row_number'])
        del preview['row_number'] # remove since not in original
        self.assertEqualDf(preview,
                           df.iloc[[0,1,2,3,4,
                                    15,16,17,18,19,
                                    20,21,22,23,24,
                                    25,26,27,28,29
                           ]])
        
if __name__=='__main__':
    sys.exit(unittest_dfx.main(__file__))
