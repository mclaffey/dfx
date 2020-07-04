#!/usr/bin/env python

import sys
import pickle
import os
import curses
import curses.textpad
import random
import textwrap
import pickle
import threading

import dfx.ops
import dfx.grain

TIMEOUT=300

pd = None # import on demand when needed
def import_pandas():
    global pd
    if pd:
        return
    import pandas
    pd = pandas
    pd.options.display.max_colwidth=12


class Environment(object):
    """Holds multiple DfViews and moves between them
    """
    def __init__(self):
        # dict of all DfViews, with keys file names or derived
        # names
        self.dfvs={}        
        # key of current Dfview
        self.current_dfv_name=None
        # directory to look for other csvs
        self.current_dir=None

    @property
    def dfv(self):
        return self.dfvs[self.current_dfv_name]

    def next(self):
        """Cycle to next dfv by alphabetical name
        """
        names = sorted(self.dfvs.keys())
        i = names.index(self.current_dfv_name)
        if i == len(names)-1:
            i = 0
        else:
            i+=1
        self.current_dfv_name = names[i]

    def next_file(self):
        """Load the next csv in current directory that isn't in dfvs
        yet

        Returns the loaded file name. If no files were found, or all
        files have already been loaded, returns None
        """
        for f in sorted([os.path.join(self.current_dir, f) for f in
                         os.listdir(self.current_dir)
                         if f.endswith('.csv')]):
            if f not in self.dfvs:
                self.load_file(f)
                return f
        return None

    def load_file(self, path):
        """Load a specific file
        """
        if path in self.dfvs:
            raise KeyError('File already loaded')
        import_pandas()
        df=pd.read_csv(path)
        self.dfvs[path]=DfView(df)
        self.current_dfv_name=path
        self.current_dir=os.path.dirname(path)

    def close(self):
        """Shut down DfView threads and save to disk
        """
        for dfv in self.dfvs.values():
            dfv.close()
        with open('env.pickle', 'wb') as f:
            pickle.dump(self, f)

    def new_dfv(self,
                df=None,
                name='',
                parent_dfv=None):
        """Add a new DfView to environment and switch to it
        """
        self.dfvs[name]=DfView(df,
                              parent_dfv=parent_dfv)
        self.current_dfv_name=name        

class IncrementalDfComp(object):
    """Compute a function on the first 10, 100, 1000 rows
    of a dataframe
    """
    def __init__(self, df, func, *args, **kwargs):
        # attributes passed in to init
        self.df=df
        self.func=func
        self.args=args
        self.kwargs=kwargs
        # attribute calculated by class
        self.results=[]
        self.message='Not started'
        self.done=False
        self._set_nrows()
        self._thread=None

        # start first calculation
        self.still_needed()

    @property
    def result(self):
        """Return the output of func() for the largest dataset
        that has been calculated so far.

        .results() is a list of outputs from func(). They go in order,
        with the first result having been calculated on the largest
        number of rows.
        """
        # if the first result hasn't been returned yet, wait for it
        if not self.results:
            self._thread.join()
        return self.results[0]

    def _set_nrows(self):
        """Determine the incremental number of rows that each dataset
        should have when passed to func

        This goes 10, 100, 1000... up to the full number of rows in
        the dataset. If the second to last value is within 80% of the
        full dataset, that value is skipped (e.g. if there are 1050
        rows in the dataset, it will go 10, 100, 1050).

        .nrows is a list which is treated as a queue. A calculation is
        started on the first row count in the list, and when that
        calculation is completed, that first row count is popped
        off. The instance is done calculating when this list is empty.
        """
        i = 1
        self.nrows=[]
        while True:
            i *= 10
            if i < (0.8*self.df.shape[0]):
                self.nrows.append(i)
            else:
                self.nrows.append(self.df.shape[0])
                return
        
    def still_needed(self):
        """Tell the instance to calculate the next longest
        increment if one is available

        The intention is that every time calling code checks the
        instance for the current result, it would call this method to
        indicate that it is still interested in a more complete
        response.

        """
        if self.done:
            return
        if self._thread and self._thread.isAlive():
            return
        self._thread=threading.Thread(target=self.calc) 
        self._thread.start()
        
    def calc(self):
        """This is the workhorse method that is executed
        as a separate thread

        The calling code isn't expected to call this method; it should
        call .still_needed() and check .done.
        """
        if not self.nrows:
            raise ValueError('No more nrows')
        nrow=self.nrows[0]
        self.message='Running for {} rows'.format(nrow)
        res = self.func(self.df[:nrow],
                        *self.args,
                        **self.kwargs)
        self.results.insert(0, res)
        self.message='Done with {} rows'.format(nrow)
        self.nrows.remove(nrow)
        if not self.nrows:
            self.done=True
            self._thread=None

    def cancel(self):
        """TODO I only need this because I'm trying to pickle a
        DfView, and that can't pickle a thread. So for now I'm just
        clearing the reference and not worrying about the thread
        continuing to run in the background.
        """
        self._thread=None

def streak_next(values, i):
    """Given a list of numbers, skip forward over consecutive streaks

    Used by DfView.find()
    
    If i references a value within a consecutive streak, the value
    returned will be the index of the end of the streak. If i references
    a value at the end of a streak, the value returned will be the
    index of the next streak
    
    Example:
        values = [-1, 3, 4, 5, 10, 11]
        i = 0                 # values[i] =-1
        i = streak_next(x, i) # values[i] = 3
        i = streak_next(x, i) # values[i] = 5
        i = streak_next(x, i) # values[i] = 10
        i = streak_next(x, i) # values[i] = 11
    """
    while i<len(values)-1:
        i+=1
        if i == len(values)-1:
            break
        prev_val=values[i-1]
        curr_val=values[i]
        if curr_val-prev_val>1:
            break
        next_val=values[i+1]
        if next_val-curr_val>1:
            break
    return i

def streak_prev(values, i):
    """Given a list of numbers, skip backwards over consecutive
    streaks.

    See streak_next(); this works in the opposite direction.
    """
    while i>0:
        i-=1
        if i == 0:
            break
        next_val=values[i+1]
        curr_val=values[i]
        if next_val-curr_val>1:
            break
        prev_val=values[i-1]
        if curr_val-prev_val>1:
            break
    return i

        
class DfView(object):

    def __init__(self, df, parent_dfv=None):

        # reduce
        df = self._strip_cols(df)
        self.reduced = dfx.ops.ReducedDf(df)
        self.df=self.reduced.df

        # add row_number and selected
        self.df=self.df.reset_index().rename(
            columns={'index': 'row_number'})
        self.df['row_number'] += 1
        #self.df['selected']=''

        """
        parent_dfv is the DfView that this one was derived from

        Created for value counts join filter
        """
        self.parent_dfv=parent_dfv

        """
        a subset of rows (Pandas dataframe) to be displayed to the
        user. Updated by generate_preview()        
        """
        self.row_preview = None

        ## column settings
        
        self.col_selected=0
        self.cols_hidden = []
        self.cols_keep_left = []

        ## row settings
        """
        _preview_head_n/_preview_tail_n
          count of rows from begininning and end of dataframe
          to be displayed in preview
        _preview_mid_n
          count of rows besides head/tails to show
          
        """
        self._preview_head_n = 5
        self._preview_tail_n = 5
        self._preview_mid_n = 10
        self._preview_center_i = int(self.df.shape[0]/2)

        """
        Which columns to sort by, and ascending/descending

        This is a list of tuples, where each tuple is a column name
        and boolean, with the boolean True for ascending and False
        for descending. Set by .sort()
        """
        self._sort_fields = []

        """
        Not sure about this design, namely where to store the
        preference to randomize the row_preview. Rather than storing
        with DfView, it might be a property of the user environment
        across all DfViews.

        Regardless, this isn't currently linked to any behavior of
        DfView; it is up to the calling code to use:
          if dfv.randomize:
            dfx.set_preview(random=True)
        
        """
        self.randomize=False

        """
        integer offset of row that user has selected with cursor
        with .row_preview (this is not a value from self.df.index)
        """
        self.row_selected=-1

        self.set_preview(randomize=True)


    """Columns added by DfView
    """
    _dfv_cols=['row_number']

    def _strip_cols(self, df=None):
        """Remove row_number and selected columns

        :df - a DataFrame, in case you want to call this on a
        dataframe different than the instance's dataframe
        """
        if df is None: df=self.df
        cols = [_ for _ in self._dfv_cols
                if _ in df.columns]
        return df.drop(columns=cols)

    """Grain computation

    Initially None. The first call to self.grain, this is populated
    by an instance of IncrementalDfComp(...GrainDf...).
    
    """
    _grain_comp = None
        
    """Grain columns, (set)

    Manual specification of grain columns. 
    
    If this is None, grain columns will be determined by GrainDf.

    If not None, the user has manually specified columns to use as
    grain. These columns will be passed to GrainDf.
    
    """
    _grain_columns = None
        
    @property
    def grain(self):
        if self._grain_comp is None:
            df = self._strip_cols()
            self._grain_comp=IncrementalDfComp(
                df,
                dfx.grain.GrainDf,
                columns=self._grain_columns,
                force=bool(self._grain_columns),
                uniq_threshold=0,
            )
        self._grain_comp.still_needed()
        return (
            self._grain_comp.result,
            self._grain_comp.done,
            self._grain_comp.message,
            )

    def add_grain_column(self, column):
        """Force a column to be part of grain definition

        Returns a newly-calculating self.grain

        If _grain_column is None when this is called, and _grain_comp
        has any columns currently, that's taken as the starting point
        and 'column' is added. The list is saved as _grain_columns,
        and then _grain_comp is recalculated.

        """
        if column not in self.df.columns:
            raise KeyError("Not a column: '{}'".format(
                column))
        if not self._grain_columns:
            if self._grain_comp is None:
                self._grain_columns = set()
            else:
                self._grain_columns =set(
                    self._grain_comp.result.columns)
        
        if column in self._grain_columns:
            return
        self._grain_columns.add(column)

        # recalc grain attributes
        self._grain_comp = None
        return self.grain

    def remove_grain_column(self, column):
        """Force a column to be removed from grain definition

        Returns a newly-calculating self.grain

        If _grain_column is None when this is called, and _grain_comp
        has any columns currently, that's taken as the starting point
        and 'column' is removed. The list is saved as _grain_columns,
        and then _grain_comp is recalculated. If column is not part of
        _grain_comp's columns, this will still have the effect of
        converting to manually specified grain column.       
        """
        if column not in self.df.columns:
            raise KeyError("Not a column: '{}'".format(
                column))
        if not self._grain_columns:
            if self._grain_comp is None:
                raise ValueError('No grain columns currently specified')
            else:
                self._grain_columns =set(
                    self._grain_comp.result.columns)                
        if column not in self._grain_columns:
            return
        self._grain_columns.remove(column)

        # recalc grain attributes
        self._grain_comp = None
        return self.grain

    def melt(self):
        """Melt to grain column        
        """
        df=self._strip_cols()
        return df.melt(id_vars=self.grain[0].columns)
    
    def pivot(self, column):
        """Pivot on a column

        Returns a new dataframe. Does NOT modified the existing
        DfView.        

        Grain columns stay as columns. The values of the pivoted
        columns turn into column groups. All remaining non-grain
        columns are taken as measures for the pivot.

        No aggregation occurs.

        :column - str. column to pivot on
        """
        if column not in self.df.columns:
            raise ValueError("Not a column: '{}'".format(column))
        id_cols = [_ for _ in self.grain[0].columns
                   if _!=column and _ not in self._dfv_cols]        

        # check if this will be unique
        max_cell_n = self.df.groupby(id_cols+[column]).size().max()
        if max_cell_n > 1:
            raise ValueError(
                'There would be more than 1 value per pivoted cell'
                ' (max {})'.format(max_cell_n))                
        
        val_cols = [_ for _ in self.df.columns
                    if _ not in id_cols+[column]+self._dfv_cols]
        piv_df = self.df.pivot_table(index=id_cols,
                                     columns=[column],
                                     values=val_cols,
                                     aggfunc='first')            
        piv_df = piv_df.reset_index()
        piv_df.columns = ['_'.join([
            col_lev for col_lev in col if col_lev])
                          for col in piv_df.columns]
        return piv_df
    

    @property
    def value_patterns(self):
        if not hasattr(self, '_vp_comp'):
            self._vp_comp=IncrementalDfComp(
                self.df, dfx.ops.ValuePatternsDf)
        self._vp_comp.still_needed()
        return (
            self._vp_comp.result,
            self._vp_comp.done,
            self._vp_comp.message,
        )

    def close(self):
        """Stop any threads on IncrementalDfComps
        """
        if hasattr(self, '_grain_comp'):
            self._grain_comp.cancel()
        if hasattr(self, '_vp_comp'):
            self._vp_comp.cancel()

    @property
    def col_widths_orig(self):
        """Calculate the maximum string length of values in each
        column, using the first and last 5 rows
        """
        if not hasattr(self, '_col_widths_orig'):
            preview_orig = self.row_preview
            mid_orig = self._preview_mid_n
            self._preview_mid_n=0
            df=self.set_preview()
            self.row_preview=preview_orig
            self._preview_mid_n=mid_orig
            
            self._col_widths_orig = {
                col_name: max([len(str(_)) for _ in col]+\
                              [len(col_name)])
                for col_name, col in df.iteritems()}
        return self._col_widths_orig

    _find_str = ''
    _find_column_name = ''
    _find_match_row_i = []
    _find_match_i = None

    @property
    def find_status(self):
        """Return a string indicating what was searched for and how
        many results were found.
        """
        if not self._find_str:
            return 'No find initiated'

        if not self._find_match_row_i:
            return "Find {}='{}', no matched.".format(
                self._find_column_name,
                self._find_str)                
        
        return "Find {}='{}', showing {} of {}".format(            
            self._find_column_name,
            self._find_str,
            self._find_match_i + 1,
            len(self._find_match_row_i))

    def find(self, find_str=None, direction=None, column=None):
        """Find text in a column

        Returns False if no match, and doesn't update
        anything. Otherwise returns True and updates row_preview.        

        find in specific column:
          find(find_str='abc', column='col1')

        repeat same find_str, but in a different column
          find(column='col23')

        repeat find, looking in columns to the right:
          find(direction='right')

        find a different string, in the same column that was searched
          last:
          find(find_str='xyz')

        reset self.row_preview to whatever the last find was. If the
        last find wasn't successful, returns False and doesn't update
        the row_preview.

        """

        #print('.find(find_str={}, direction={}, column={}'.format(
        #    find_str, direction, column))

        if not find_str and not direction and not column:
            raise ValueError('Must provide either find_str, direction'
                             'or column')
        #if find_str and direction: error?
        #if find_str and not column: must provide column (for now)

        # are we looking in a new column?
        new_column_for_find=False
        if column and self._find_column_name!=column:
            new_column_for_find=True
            self._find_column_name=column
            
        # if column isn't specified, use first column
        if not self._find_column_name:
            self._find_column_name=self.row_preview.columns[0]

        # are we looking for a new string?
        new_str_for_find=False
        if find_str and self._find_str!=find_str:
            new_str_for_find=True            
            self._find_str=find_str
        #else:
            #print('not new find_str?', find_str, self._find_str,
            #find_str=self._find_str)

        # if we have new string or column, execute find
        if new_column_for_find or new_str_for_find:
            col=self.df[self._find_column_name].apply(str).reset_index(drop=True)            
            row_i=col.str.contains(self._find_str, case=False)
            if not row_i.any():
                self._find_match_row_i=[]
                self._find_match_i=None
                #print('.find() execute - no matches')
                return False
            self._find_match_row_i=row_i.index[row_i].array
            self._find_match_i=0
            #print('.find() execute: {} matches'.format(
                #len(self._find_match_row_i)))
            # we'll update self.row_preview at the end of this method

        # direction logic
        if direction=='next':
            self._find_match_i += 1
            n = len(self._find_match_row_i)
            if self._find_match_i == n:
                self._find_match_i = n-1
        if direction=='previous':
            self._find_match_i -= 1
            if self._find_match_i < 0:
                self._find_match_i = 0
        if direction=='down':    
            self._find_match_i = streak_next(
                self._find_match_row_i,
                self._find_match_i)
        if direction=='up':
            self._find_match_i = streak_prev(
                self._find_match_row_i,
                self._find_match_i)
        if direction=='left':
            cols = list(self.row_preview.columns)
            col_i = cols.index(self._find_column_name)
            col_i-=1            
            while col_i >= 0:                
                new_col = self.df.columns[col_i]
                if self.find(column=new_col):
                    # at this point, the above find command has found
                    # a match and set the row_preview, so we can
                    # return True for the match without doing anything
                    # else
                    return True
                col_i-=1                
            if col_i<0:
                # if we we all the way to the left with no matches,
                # return False
                return False

        if direction=='right':
            cols = list(self.row_preview.columns)
            cols_n = len(cols)
            col_i = cols.index(self._find_column_name)
            col_i+=1
            while col_i < cols_n:
                new_col = self.df.columns[col_i]
                if self.find(column=new_col):
                    # at this point, the above find command has found
                    # a match and set the row_preview, so we can
                    # return True for the match without doing anything
                    # else
                    return True
                col_i+=1
            if col_i >= cols_n:
                # if we we all the way to the rightwith no matches,
                # return False
                return False

        # update preview
        # we could reach this point because either:
        #   - new find string
        #   - new find column
        #   - direction command
        #print('.find() updating preview')

        # if we are here without any matches, don't update row_preview
        # and return False
        if self._find_match_i is None:
            return False

        # otherwise update row_preview to current find
        center_i=self._find_match_row_i[self._find_match_i]
        self.set_preview(center_i=center_i)
        self.col_selected=list(self.row_preview.columns).index(
            self._find_column_name)
        return True
            
    

    def sort(self, field, ascending=True, add=False):
        """Sort the dataframe for display in row_preview

        This method is called for one field at a time. Use add=True
        to add a second, third, etc sort field.

        :field - string. Name of column to sort by. If None,
          existing fields are discarded and sort is set to
          'row_number' ascending.        
        :ascending - boolean, default True. If True, sort
          ascending. If False, sort descending.
        :add - boolean, default False. If False, any existing
          sort fields will be discarded and new sort is only based
          on 'field'. If True, 'field' is added as a sort field
          after any existing fields.
        """
        self.randomize=False

        # no sort means sort by row number
        if field is None:
            self._sort_fields=[]
            self.df.sort_values('row_number', inplace=True)
            self.set_preview()
            return

        # set self._sort_fields
        if not add:
            self._sort_fields = [(field, ascending)]
        else:
            # remove if already in it
            for sort_field in self._sort_fields:
                if sort_field[0]==field:
                    self._sort_fields.remove(sort_field)
                    break
            self._sort_fields.append((field, ascending))

        # perform sort
        fields, ascendings = zip(*self._sort_fields)
        self.df.sort_values(list(fields),
                            ascending=ascendings,
                            inplace=True)

        # update preview
        self.set_preview()
    
    def set_preview(self,
                    center_i=None,
                    offset=None,
                    randomize=False,
    ):
        """Get a set of rows from the dataframe

        Typically this will be the first few rows, the last few rows,
        and sample of rows in between. The middle sample can either be
        set by specifying a row to center on (center_i=), by
        specifying a number of rows to move up or down (offset=), or
        by specifying self.randomize=True and a random sample will be
        selected.
        
        If center_i or offset are provided, the relevant rows are
        set. If these are not provided, the mid rows are selected
        by random sample.

        Only one of the following can be specified at a time:
          - center_i
          - offset
          - randomize
        
        :center_i - int. Optional, a row number. If provided, the
           mid_rows are selected so that they center on the specified
           row. First row is 1, not 0.
        :offset - int, default None. If not None and positive int,
           shift the middle rows down to show later row numbers. If
           negative, shift up.
        :randomize - bool. Default False. If True, middle rows
           will be randomly sampled from dataframe, but in order.

        """

        # check arguments
        arg_sum = sum([1 if _ else 0 for _ in
                       [randomize, center_i, offset]])
        if arg_sum > 1:        
            raise ValueError(('Can not specify more than one: '+\
                              'randomize={} '.format(randomize)+\
                              'center_i={} '.format(center_i)+\
                              'offset={}'.format(offset)))

        # if offset, call set_preview() again
        if offset is not None:
            return self.set_preview(
                center_i=self._preview_center_i+offset)
        
        # calculate some stuff
        nrow=self.df.shape[0]        # count of df rows
        head_n=self._preview_head_n
        tail_n=self._preview_tail_n
        head_i = list(range(head_n)) # indices of first rows to show
        tail_i = sorted([nrow - i - 1 for i in range(0, tail_n)])
        # get count for middle, unless dataframe is smaller than that
        mid_n = min(self._preview_mid_n, nrow-head_n-tail_n)

        # arrange columns
        show_cols = self.cols_keep_left +\
                    [_ for _ in self.df.columns
                     if _ not in self.cols_hidden 
                     and _ not in self.cols_keep_left]
        df = self.df[show_cols]
        
        # if small dataset, return as is
        if nrow <= (head_n+tail_n):
            self.row_preview = df
            return self.row_preview
           
        # if not asking for middle rows, return head+tail
        if mid_n <= 0:
            self.row_preview=df.iloc[head_i+tail_i]
            return self.row_preview

        # if no arguments, just refreshing
        if arg_sum == 0:
            if self.randomize:
                randomize=True
            else:
                center_i = self._preview_center_i

        # if center_i is being specified, grab rows before/after
        # it and updated row_selected
        if center_i is not None:
            # if more than number of rows, cap to last row
            if center_i >= nrow:
                center_i = nrow-1
            self._preview_center_i=center_i
            if center_i in head_i:
                self.row_selected = head_i.index(center_i)
                return self.row_preview
            elif center_i in tail_i:
                self.row_selected = head_n+mid_n+tail_i.index(center_i)-1
                return self.row_preview
            elif center_i >= (nrow - mid_n - tail_n):
                mid_i = list(range(nrow-mid_n-tail_n, nrow))
                self.row_selected = head_n + mid_i.index(center_i)
            elif center_i < (head_n + mid_n):
                mid_i = list(range(head_n, head_n+mid_n))
                self.row_selected = head_n + mid_i.index(center_i)
                
            else:
                start_i = center_i - int(mid_n/2)
                mid_i = list(range(
                    start_i,
                    start_i + mid_n,
                ))
                self.row_selected = head_n + int(mid_n/2)

        # random, grab that many random rows
        if randomize:
            mid_i=random.sample(
                range(head_n, nrow-tail_n),
                mid_n)
            rows_i = sorted(mid_i)
            # row_selected isn't updated. the rows on the screen will
            # be changing, but the cursor will stay in the same place

        # use mid_i to set row_preview
        rows_i = sorted(set(head_i + mid_i + tail_i)) # dedup
        #try:
        #    rows_i = set(head_i + mid_i + tail_i) # dedup
        #except UnboundLocalError as e:
        #    raise ValueError(center_i, head_i, tail_i, center_i in head_i)
        rows_i = list(filter(lambda x: 0 <= x < nrow, rows_i)) # cap
        self.row_preview=df.iloc[rows_i]
        return self.row_preview

    @property
    def reduced_short_desc(self):
        """Provide a brief (one line?) summary of reduced
        """
        if not self.reduced.reduced:
            return 'Not reduced'
        s = ""

        if self.reduced.constants:            
            s += 'Constants: ' + ",".join([
                '{}={:30.30s}'.format(k,str(v))
                for k,v in self.reduced.constants.items()])
        if self.reduced.zeros:
            s += " Zeros: " + ",".join(self.reduced.zeros)
        if self.reduced.nulls:
            s += " Nulls: " + ",".join(self.reduced.nulls)
        s = s.strip()

        return s

def rect(s):
    return "\n".join([_[:120] for _ in s.split("\n")[:40]])
        
def curses_loop(scr, env):
    key = ''
    scr.timeout(TIMEOUT)
    find_mode=False
    find_value=None
    jump_mode=False
    jump_value=''
    message=''

    while key != 'q':

        dfv=env.dfv
        df=dfv.df

        # row preview
        if dfv.randomize:
            dfv.set_preview(randomize=True)
        dfi=dfv.row_preview
        
        # columns to be displayed
        col_names = dfi.columns
        curr_column_name = col_names[dfv.col_selected]
            
        ### calculate column widths
        x_max = 140  # width of screen
        y_max = 45
        col_spacer=3 # spaces between columns
        col_min=3    # show atleast this many chars per column
        # initial widths based on longest values
        col_widths = {k:v+col_spacer for k,v in
                      dfv.col_widths_orig.items()                      
                      if k in col_names}        
        # keep reducing the longest column until they all fit or
        # are at the minimum width
        while sum(col_widths.values()) > x_max:
            col_max_name = ''
            col_max_width = 0
            for col_name, col_width in col_widths.items():
                if col_width > col_max_width:
                    col_max_width = col_width
                    col_max_name = col_name
            col_widths[col_max_name] -= 1
            if col_max_width==col_min+col_spacer:
                break
        col_widths = {k:v-col_spacer for k,v in col_widths.items()}

        # column headers
        col_headers = {}
        col_head_rows=3
        for col_name in col_names:
            col_head_strs = textwrap.wrap(col_name,
                                          width=col_widths[col_name])
            while len(col_head_strs) < col_head_rows:
                col_head_strs.insert(0, '')
            col_head_strs = col_head_strs[:col_head_rows]
            col_headers[col_name] = col_head_strs

        # column footers (value patterns)
        col_footers = {}
        col_foot_rows=3
        vp_result, vp_done, vp_message = dfv.value_patterns
        if not vp_result:
            col_footers = {_: ['']*col_foot_rows for _ in col_names}
        else:
            for col_name in col_names:
                if col_name in vp_result.value_patterns:
                    vp = vp_result.value_patterns[col_name][0]                    
                    vp_str = ", ".join(vp.keys())
                else:
                    vp_str='NA'
                col_foot_strs = textwrap.wrap(vp_str,
                                              width=col_widths[col_name])
                col_foot_strs = col_foot_strs[:col_foot_rows]
                col_footers[col_name] = col_foot_strs
                
        # print header to screen
        scr.erase()
        y=0

        # screen line - which dfv
        line='{shape:15s} {dfv_name:} > {col_name:}({col_i:})'.format(
            shape=str(df.shape),
            dfv_name=env.current_dfv_name,
            col_name=col_names[dfv.col_selected],
            col_i=dfv.col_selected)
        scr.addstr(y, 0, line)
        y+=1

        # message
        scr.addstr(y, 0, message)
        y+=1
        
        # screen line - key, randomize on, find
        line='{:15s} {} {}'.format(
            key,
            'RANDOMIZE' if dfv.randomize else 'not randomize',
            'Jump {}={}'.format(jump_mode, jump_value),
            )
        scr.addstr(y, 0, line)
        y+=1

        # find
        if not find_mode:
            find_str=''
        elif find_mode=='edit':
            find_str='Find: {}_'.format(find_value)
        else:
            find_str=dfv.find_status
        scr.addstr(y, 0, find_str)
        y+=1
            
        # screen line - sort
        scr.addstr(y, 0, 'Sort: {}'.format(dfv._sort_fields))
        y+=1

        # screen line - reduced
        scr.addstr(y, 0, "Reduced: {}".format(dfv.reduced_short_desc)[:x_max])
        y+=1

        # screen line - value patterns
        scr.addstr(y, 0, "Value pattern: {}".format(vp_message))
        y+=1

        # screen line - grain
        grain_result, grain_done, grain_message = dfv.grain
        if not grain_result:
            grain_str = grain_message
        else:
            if grain_done:
                grain_message = ''
            else:
                grain_message = ' ({})'.format(grain_message)
            grain_str = 'Grain{}: {}'.format(
                grain_message,
                ". ".join(_.strip() for _ in
                          grain_result.summary.split('\n')).replace('  ', ' '))
        scr.addstr(y, 0, grain_str[:x_max])
        y+=1

        # print dataframe to screen
        x = 0
        data_top_y=y
        for col_i, col_name in enumerate(col_names):
            y = data_top_y
            col_w = col_widths[col_name]
            col_w = min(col_w, x_max-x) # not past right edge
            
            # column heading
            addstr_args=[]
            if col_i==dfv.col_selected:
                addstr_args.append(curses.A_REVERSE)
            for i, s in enumerate(col_headers[col_name]):
                scr.addstr(y+i, x, s[:col_w], *addstr_args)
            y+=col_head_rows

            # heading underling
            scr.addstr(y, x, '-'*col_w)
            y+=1

            # column values
            col = dfi[col_name]
            for i, val in enumerate(col):
                if y+i > y_max:
                    raise ValueError(y+i, dfi.shape)
                addstr_args=[]
                if i==dfv.row_selected and col_i==dfv.col_selected:
                    addstr_args.append(curses.A_REVERSE)
                scr.addstr(y+i,
                           x,
                           str(val)[:col_w],
                           *addstr_args)
            y+=col.size
                
            # footer underling
            scr.addstr(y, x, '-'*col_w)
            y+=1

            # column footers (value patterns)
            for i, s in enumerate(col_footers[col_name]):
                addstr_args=[]
                if col_i==dfv.col_selected:
                    addstr_args.append(curses.A_REVERSE)
                scr.addstr(y+i, x, s[:col_w], *addstr_args)
            y+=col_foot_rows
                       
            # increment for next column, but stop if already wide enough
            x+=col_w + col_spacer
            if x > x_max:
                break

        # command footer
        cmd_ftr="""
 a A       sort ascending   l         load file                 m         melt                  s         selected on/off
 z Z       sort descending  g G       add to grain (G=remove)   n         next DfView           t         filter join to parent
 c         value counts     ^g        grain details             p         add column to pivot   u         scroll up
 d         scroll down      h         unhide columns            r         reduced detail        v         value patterns
 f         find             j         jump to row number        ^R        randomize on/off      y         unsort (row_number)
 <arrows>  move cursor      <enter>   filter to value           <space>   hide column           <         freeze column to left 
"""
        if find_mode=='edit':
            cmd_ftr="""
<enter>    find
"""
        if find_mode=='command':
            cmd_ftr="""
 u  skip up   (stay in column)    p  previous  step (will go to previous column)    l  left      search columns to left
 d  skip down (stay in column)    n  next      step (will go to next column)        r  right     search columns to right  

 f            to edit mode                          q exit
 <enter>      to edit mode
 <delete>     to edit mode, delete last char
"""
        y+=1
        scr.addstr(y, 0, cmd_ftr.strip('\n'))
        y+=1
            
        # get key and navigate
        try:
            key = scr.getkey()
        except curses.error:
            key = ''

        # find mode
        if find_mode and key != '':
            if find_mode=='edit':
                if key=='\n':
                    if find_value=='':
                        find_mode=False
                    else:
                        find_mode='command'
                elif key=='KEY_BACKSPACE':
                    if find_value:
                        find_value=find_value[:-1]
                elif key.startswith('KEY_'):
                    pass    
                else:
                    find_value+=key
                # if there are atleast 3 chars, or user just
                # hit enter, find
                if len(find_value) >= 2 or find_mode=='command':
                    dfv.find(find_str=find_value,
                             column=curr_column_name)                    
                    
            elif find_mode=='command':
                if key=='n':
                    dfv.find(direction='next')
                if key=='p':
                    dfv.find(direction='previous')
                if key=='d':
                    dfv.find(direction='down')
                if key=='u':
                    dfv.find(direction='up')
                if key=='l':
                    dfv.find(direction='left')
                if key=='r':
                    dfv.find(direction='right')
                if key=='f' or key=='\n':
                    find_mode='edit'
                if key=='KEY_BACKSPACE':
                    find_mode='edit'
                    if find_value:
                        find_value=find_value[:-1]
                    if find_value:
                        dfv.find(find_str=find_value,
                             column=curr_column_name)                    
                
                if key=='q':
                    find_mode=False
            else:
                raise ValueError(
                    'Bad find mode: {}'.format(find_mode))
            if key not in ['KEY_UP', 'KEY_DOWN', 'KEY_LEFT',
                'KEY_RIGHT']:                
                key=''

        # jump mode
        if jump_mode and key != '':
            if key=='\n':
                jump_mode=False
                jump_value=''
            if key=='KEY_BACKSPACE':
                if jump_value:
                    jump_value=jump_value[:-1]                
            if key in '0123456789':
                jump_value+=key
                
            key=''                
            jump_int=None
            try:
                jump_int = int(jump_value)
            except ValueError:
                pass
            if jump_int is not None and jump_int != 0:
                jump_int -= 1
                dfv.set_preview(center_i=jump_int)

        # move around
        if key=='KEY_LEFT':
            if dfv.col_selected > 0:
                dfv.col_selected -= 1
        if key=='KEY_RIGHT':
            if dfv.col_selected < len(col_names)-1:
                dfv.col_selected += 1
        if key=='KEY_DOWN':
            dfv.row_selected+=1
        if key=='KEY_UP':
            dfv.row_selected-=1

        # a A - sort ascending
        if key=='a':
            col_name=col_names[dfv.col_selected]
            dfv.sort(col_name, add=True)
        if key=='A':
            col_name=col_names[dfv.col_selected]
            dfv.sort(col_name, add=False)
            
        # c - value counts
        if key=='c':
            vc_df = df[curr_column_name].value_counts()\
                                        .to_frame().reset_index()
            vc_df.columns = [curr_column_name, 'count']
            env.new_dfv(df=vc_df,
                        name=env.current_dfv_name + ' > counts',
                        parent_dfv=dfv)

        # d - scroll down
        if key=='d':
            dfv.set_preview(offset=1)

        # f F - find
        if key=='f':
            find_mode='edit'
            find_value=''

        # g G - add/remove from grain columns
        if key=='g':
            dfv.add_grain_column(curr_column_name)
        if key=='G':
            dfv.remove_grain_column(curr_column_name)
            
        # ^g - show GrainDf details
        if key=='^g':
            scr.erase()
            scr.addstr(0,0,dfv.grain.summary)
            scr.getkey()

        # h - unhide columns
        if key=='h':
            dfv.cols_hidden=[]

        # j - jump to row
        if key=='j':
            dfv.randomize=False
            jump_mode=True

        # l - load file
        if key=='l':
            env.next_file()

        if key=='m':
            new_df = dfv.melt()
            new_dfv=DfView(new_df)
            new_dfv_name=env.current_dfv_name +\
                          ' > melt'
            env.dfvs[new_dfv_name]=new_dfv
            env.current_dfv_name=new_dfv_name            

        # n - next DfView
        if key=='n':
            env.next()

        # p - pivot on column
        if key=='p':
            piv_col=curr_column_name
            try:
                new_df=dfv.pivot(piv_col)
            except ValueError as e:
                message=e.args[0]
                piv_col=None
                
            if piv_col:                
                new_dfv_name = env.current_dfv_name +\
                               ' > pivot ' + piv_col
                new_dfv=DfView(new_df)
                env.dfvs[new_dfv_name] = new_dfv
                env.current_dfv_name=new_dfv_name
            
        # r - show ReducedDf details
        if key=='r' and dfv.reduced.reduced:
            scr.erase()
            y=0
            scr.addstr(y, 0, 'Reduced on {}'.format(env.current_dfv_name))
            y+=2
            for k,v in dfv.reduced.constants.items():
                scr.addstr(y, 0, '{:>20s} = {}'.format(k,v)[:x_max])
                y+=1
            if dfv.reduced.zeros:
                y+=1
                scr.addstr(y, 0, 'Zeros:')
                y+=1
                for col in dfv.reduced.zeros:
                    scr.addstr(y, 2, col)
            if dfv.reduced.nulls:
                y+=1
                scr.addstr(y, 0, 'Nulls:')
                y+=1
                for col in dfv.reduced.nulls:
                    scr.addstr(y, 2, col)
            scr.timeout(0)
            scr.getkey()
            scr.timeout(TIMEOUT)

        # ^R - toggle randomization
        if key=='^R':
            dfv.randomize=not dfv.randomize

        # s - toggle selected
        if key=='s':
            if 'selected' not in df:
                df['selected'] = ''
                # kludge: force recalc of column widths
                delattr(dfv, '_col_widths_orig')
            row_i = dfv.row_preview.index[dfv.row_selected]
            new_value='x' if df.loc[row_i, 'selected']=='' else ''
            df.loc[row_i, 'selected']=new_value
            dfv.set_preview()

        # t - filter selected back to parent
        if key=='t':
            par_df=dfv.parent_dfv.df
            df = df[df.selected=='x'].copy()
            if 'row_number' in df:
                del df['row_number']
            if 'count' in df:
                del df['count']
            del df['selected']
            new_df=par_df.merge(df, how='inner')
            del new_df['row_number']
            env.new_dfv(df=new_df,
                        name = ' > multi-filter',
                        parent_dfv=dfv.parent_dfv
                        )        
            
        # u - scroll up
        if key=='u':
            dfv.set_preview(offset=-1)

        # v - show ValuePatternsDf details
        if key=='v':
            scr.erase()
            scr.addstr(0,0, str(dfv.value_patterns))
            scr.timeout(0)
            scr.getkey()
            scr.timeout(TIMEOUT)

        # y - remove sort
        if key=='y':
            dfv.sort(None)

        # z Z - sort descending
        if key=='z':
            col_name=col_names[dfv.col_selected]
            dfv.sort(col_name, ascending=False, add=True)
        if key=='Z':
            col_name=col_names[dfv.col_selected]
            dfv.sort(col_name, ascending=False, add=False)

        # ENTER, filter to value
        if key=='\n':
            col_name=col_names[dfv.col_selected]
            filter_val=dfi[col_name].values[dfv.row_selected]
            df_filt=df[df[col_name]==filter_val]
            dfv=DfView(df_filt)
            env.current_dfv_name=env.current_dfv_name +\
                ' > {}={}'.format(col_name, filter_val)
            env.dfvs[env.current_dfv_name]=dfv

        # hide column
        if key==' ':
            dfv.cols_hidden.append(col_names[dfv.col_selected])
            if dfv.col_selected == len(col_names)-1:
                dfv.col_selected -= 1

        # freeze column to left
        if key=='<':
            dfv.cols_keep_left.append(col_names[dfv.col_selected])

                
    # save state
    env.close()
                


def main():

    # if environment exists
    env_path = 'env.pickle'
    if os.path.exists(env_path):
        with open(env_path, 'rb') as f:
            env = pickle.load(f)
        res = curses.wrapper(curses_loop, env)
        print(res)
        sys.exit()

    # otherwise build environment from command line args, if any
    env = Environment()

    # determine path to start loading files from    
    if sys.argv[1:]:
        start_path = sys.argv[1]
    else:
        start_path = os.getcwd()
    if os.path.isfile(start_path):
        env.load_file(start_path)
    else:
        env.current_dir = start_path
        if not env.next_file():
            return 'No csvs found in {}'.format(
                os.path.abspath(env.current_dir))

    # run main loop    
    res = curses.wrapper(curses_loop, env)
    print(res)

if __name__ == '__main__':
    sys.exit(main())
    
