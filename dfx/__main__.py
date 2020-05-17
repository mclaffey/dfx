#!/usr/bin/env python

import sys
import os
import inspect
import builtins

import dfx.ops

# import pandas as pd # imported when needed below

def main(args=None, print_func=None):
    """Entry point for command line

    :args - list of str, as in sys.argv
    :print_func - optional, callable. If provided, output
       is passed to this function instead of print(). Used
       for testing.
    """
    if not args:
        args = sys.argv
    
    if print_func:
        print=print_func
    else:
        print=builtins.print

    ### usage
    this = os.path.basename(args.pop(0))
    cl_hint=''
    if this=='__main__.py':
        this='python -m dfx'
        cl_hint="\n\nThis can also be called with 'dfx [...]'"
    usage="""Command line entry for dfx

  > {this:} my_data.csv                     # show column types on dataset
  > {this:} my_data.csv my_col              # show detail on that column    
  > {this:} my_data.csv my_col norm         # show num_normal pattern for that column
  > {this:} my_data.csv my_col norm source  # show class/file for rule
  > {this:} --test                          # run tests
  > {this:}                                 # this screen{hint:}""".format(
      this=this, hint=cl_hint)      
    if not args or any([_ in args for _ in ['-h', '-?', 'help']]):
        print(usage)
        return

    ### run tests
    if args[0] == '--test':
        args.pop(0)
        test_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'tests')
        test_file = os.path.join(
            test_dir, 'test_ops.py')
        sys.path.append(test_dir)
        import unittest_dfx
        import unittest
        unittest.main(test_file)
        return
        unittest_dfx.main(test_file)
        return

    ### get value patterns for a dataset
    
    file_path = args.pop(0)
    if not os.path.isfile(file_path):
        return ('The file path you gave me doesnt '
                'appear to be a file: {}'.format(file_path))

    print('Importing pandas...', end='')
    import pandas as pd
    print('importing {}...'.format(file_path), end='')
    df = pd.read_csv(file_path)

    # if no args, print all value patterns and return
    if not args:
        print('analyzing...')
        dfx.ops.ValuePatternsDf(df, display=True)
        return

    ### from here on, the user has provided a column filter,
    ### so provide column-level detail
    
    col_arg = args.pop(0)
    col_matches = [col for col in df.columns
               if col_arg.lower() in col.lower()]
    if not col_matches:
        print('Columns in data set: {}'.format(df.columns))
        return 'No columns match for {}'.format(col_arg)

    df = df[col_matches]
    print('analyzing...')
    vp = dfx.ops.ValuePatternsDf(df)

    # check for source argument
    show_source = False
    if '--source' in args:
        show_source=True
        args.remove('--source')    
    
    # if arg, it's a value pattern id. also check for show_source arg
    val_pat_arg = ''
    if args:
        val_pat_arg = args.pop(0)

    for col_name, col_patterns in vp.value_patterns.items():
        print(col_name)
        for group in col_patterns:
            for val_pat_name, val_pat in group.items():
                if val_pat_arg=='' or val_pat_arg.lower() in val_pat_name.lower():
                    if show_source:
                        val_pat_class = val_pat.__class__
                        val_pat_file = inspect.getfile(val_pat_class)
                        print('From class {} in {}:'.format(
                            val_pat_class.__name__, val_pat_file))
                    print(val_pat)


        
if __name__=='__main__':
    sys.exit(main())
