#!/usr/bin/env python
import os
import sys
import unittest

import unittest_dfx

sys.path.append('..')
import dfx.ui_curses
import dfx.datasets

class IncCompTest(unittest_dfx.AbstractDfTestCase):
    """Test functionality of ui_curses.IncrementalDfComp
    """

    def test_placeholder(self):
        self.fail('No yet written')
        
if __name__=='__main__':
    sys.exit(unittest_dfx.main(__file__))
