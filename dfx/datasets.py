"""
Sample datasets used for testing/demos

02/24/2020 - combined datatsets.py from Feb 02 and 10
"""

import pandas as pd
"""
employees

  unique on employee_id
"""
employees = pd.DataFrame(
    columns=['employee_id', 'region', 'state', 'salary',
             'company', 'manager_id'],
    
    data = [
        [ '12345', 'east', 'NY', 100, 'Acme', '36363'],
        [ '24543', 'east', 'NY', 110, 'Acme', '36363'],
        [ '36363', 'east', 'NC', 120, 'Acme', '76576'],
        [ '48436', 'east', 'NC',  90, 'Acme', '36363'],
        [ '54664', 'east', 'NY', 100, 'Acme', '99999'],
        [ '69983', 'west', 'AZ', 125, 'Acme', '98765'],
        [ '76576', 'west', 'CA', 500, 'Acme', None   ],
        [ '87635', 'west', 'CA', 220, 'Acme', '98765'],
        [ '98765', 'west', 'AZ', 360, 'Acme', '76576'],
        [ '00000', 'west', 'NM', 420, 'Acme', '98765'],
    ])

""""
employees_redundant

  like employees, but a bunch of redudant fields on region
"""
_region_redundant = pd.DataFrame(
    columns=['region', 'region_address', 'region_sales'],
    data = [
        [ 'east',
          '123 Avenue of the Boulevards, Westheightsville, XO',
          '1 million dollars'],
        [ 'west',
          'Via de la Villa Caminitos, La Jolla, CA',
          '$3 unicorns']
        ])
employees_redundant = employees.merge(_region_redundant)
        




"""
employee_dups

  two duplicate IDs: 123, 456
"""
def make_employee_dups():
    df = employees.copy()
    df['status'] = 'active'
    df2 = df[df.employee_id.isin(['12345', '24543'])].copy()
    df3 = df[df.employee_id.isin(['12345'])].copy()
    df = pd.concat([df, df2, df3])
    return df
employee_dups = make_employee_dups()

"""
employee_hist

  'sparse'

  unique on employee_id, status
  status vals: active, old
"""
def make_employee_hist():
    df = employees.copy()
    df['status'] = 'active'
    df2 = df[df.employee_id.isin(['12345', '24543'])].copy()
    df2.status = 'old'
    #df2.salary = df2.salary * .9
    df = pd.concat([df, df2])
    return df
employee_hist = make_employee_hist()

"""
regions

  'hier'

  I don't think this works the way I want it to. If it's 1:many,
  1:many, I think that means county is unique on it's own, and it's
  not a multi-col ID. But it feels like I'm missing an opportunity to
  informally include state/county in the grain, such as the ability to
  use filter_up.

  unique on state, county, town
  not perfect
  1:many state  -> county
  1:many county -> town
"""
regions = pd.DataFrame(
    columns=['state', 'county', 'town'],
    data = [
        'ny monroe pittsford'.split(),
        'ny monroe penfield'.split(),
        'ny allegany belfast'.split(),
        'ny allegany wellsville'.split(),
        'ca sandiego sandiego'.split(),
        'ca sandiego chulavist'.split(),
        'fl monroe keywest'.split(),
        'fl monroe marathon'.split(),
        'vt rutland pittsford'.split(),
     ])

"""
cube

  'cube'

  unique on dept, status, date
  perfect
  many:many  dept   -> status
  many:many  dept   -> date
  many:many  status -> date

"""
def make_cube():
    rows = []
    for dept in ['finance', 'hr', 'legal']:
        for status in ['active', 'archive', 'review']:
            for date in ['01-01-2020', '02-01-2020', '03-01-2020']:
                rows.append((dept, status, date))
    return pd.DataFrame(
        columns = ['dept', 'status', 'date'],
        data = rows)            
cube = make_cube()

"""
???

  unique on a, b, c
  not perfect
  1:many     a -> b
  1:many     a -> c
  many:many  b -> c


"""

pay_periods = pd.DataFrame(
    columns=['pay_date', 'tax_rate', 'company'],

    data=[
        ['2019-01-01', .21, 'Acme'],
        ['2019-02-01', .21, 'Acme'],
        ['2019-03-01', .28, 'Acme'],
    ])

checks = employees.merge(pay_periods, on='company')
