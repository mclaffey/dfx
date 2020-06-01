#!/usr/bin/env python
"""Navigate static structure of screens
"""
import sys
import pickle
import os
import curses
import random
import textwrap
import pickle

import pandas as pd
pd.options.display.max_colwidth=12

class Environment(object):
    def __init__(self):
        self.dfvs={}
        self.df_path=''

class DfView(object):
    def __init__(self, df):
        self.df=df
        self.col_selected=0
        self.cols_hidden = []
        self.cols_keep_left = []

def rect(s):
    return "\n".join([_[:120] for _ in s.split("\n")[:40]])
        
def curses_loop(scr, env):
    key = ''
    dfv = env.dfvs[env.df_path]

    while key != 'q':

        df=dfv.df
        
        # determine rows to present
        mid = random.sample(range(5, df.shape[0]-10), 20)
        rows = list(range(5)) + mid + list(range(-5,0))
        dfi = df.iloc[rows]

        # calculate df column widths of head
        # TODO: this can be done once as part of DfView
        df_ht = df.iloc[list(range(5)) + list(range(-5,0))]
        col_widths_orig = {}
        for col_name, col in df_ht.iteritems():
            col_widths_orig[col_name] = max([len(str(_)) for _ in col])

        # columns to be displayed
        col_names = dfv.cols_keep_left +\
                    [_ for _ in df.columns
                     if _ not in dfv.cols_hidden 
                     and _ not in dfv.cols_keep_left]
            
        ### calculate column widths
        x_max = 140  # width of screen
        col_spacer=3 # spaces between columns
        col_min=3    # show atleast this many chars per column
        # initial widths based on longest values
        col_widths = {k:v+col_spacer for k,v in
                      col_widths_orig.items()                      
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
                
        # print header to screen
        scr.erase()
        scr.addstr(0, 0, key)
        scr.addstr(0, 10, str(dfv.col_selected))
        scr.addstr(0, 20, col_names[dfv.col_selected])
        scr.addstr(0, 50, env.df_path)
        scr.addstr(1, 0, str(dfv.cols_keep_left))        

        # print data to screen
        y = 2
        x = 0
        for col_i, col_name in enumerate(col_names):
            col_w = col_widths[col_name]
            col_w = min(col_w, x_max-x) # not past right edge
            # column heading
            addstr_args=[]
            if col_i==dfv.col_selected:
                addstr_args.append(curses.A_REVERSE)
            for i, s in enumerate(col_headers[col_name]):
                scr.addstr(y+i, x, s[:col_w], *addstr_args)
            scr.addstr(y+col_head_rows, x, '-'*col_w)

            # column values
            col = dfi[col_name]
            for i, val in enumerate(col):
                scr.addstr(y+col_head_rows+i+1, x, str(val)[:col_w])
            x+=col_w + col_spacer
            if x > x_max:
                break


        # get key and navigate
        key = scr.getkey()
        if key=='KEY_LEFT':
            if dfv.col_selected > 0:
                dfv.col_selected -= 1
        if key=='KEY_RIGHT':
            if dfv.col_selected < len(col_names)-1:
                dfv.col_selected += 1
        if key==' ':
            dfv.cols_hidden.append(col_names[dfv.col_selected])
            if dfv.col_selected == df.shape[1]-1:
                col_selected -= 1
        if key=='<':
            dfv.cols_keep_left.append(col_names[col_selected])
        if key=='c':
            vc=df[col_names[dfv.col_selected]].value_counts().head(40)
            s = rect(str(vc))
            scr.erase()
            scr.addstr(3,0,s)
            scr.getkey()

        if key=='f':
            df_dir = os.path.dirname(env.df_path)
            files = [os.path.join(df_dir, f) for f in
                     os.listdir(df_dir)
                     if f.endswith('.csv')]                     
            i = files.index(env.df_path)
            if i == len(files)-1:
                i = 0
            else:
                i+=1
            env.df_path = files[i]
            if env.df_path not in env.dfvs:
                env.dfvs[env.df_path] = DfView(pd.read_csv(env.df_path))
            dfv=env.dfvs[env.df_path]
            
    # save state
    with open('env.pickle', 'wb') as f:
        pickle.dump(env, f)
                


def main():
    # if environment exists
    env_path = 'env.pickle'
    if os.path.exists(env_path):
        with open(env_path, 'rb') as f:
            env = pickle.load(f)
        res = curses.wrapper(curses_loop, env)
        print(res)
        sys.exit()

    df_path = sys.argv[1]
    if os.path.isdir(df_path):
        df_dir=df_path
        csv_paths = [file_name for file_name in os.listdir(df_path)
                     if file_name.endswith('.csv')]
        if not csv_paths:
            print('No csvs found in {}'.format(df_path))
        else:
            df_path=os.path.join(df_path, csv_paths[0])
    else:
        df_dir=os.path.dirname(df_path)
    print('importing...', end='', flush=True)
    
    df = DfView(pd.read_csv(df_path))
    print('done')

    env = Environment()
    env.df_path = df_path
    env.dfvs[df_path] = df
    res = curses.wrapper(curses_loop, env)
    print(res)

if __name__ == '__main__':
    main()
    
