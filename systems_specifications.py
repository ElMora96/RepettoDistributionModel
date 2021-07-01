# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:49:56 2021

@author: giamm
"""

import pandas as pd

'''
This module contains a method to restore default values for technologies
parameters (such as efficiencies, etc.), which are saved as csv files
'''

def storage_system_specifications(default_flag=False):
    '''
    The method creates csv files containing default values for technologies
    specifications.
    
    Input:
        - default_flag, bool, if True, default values are used and restored
        
    Output:
        - storage_system_specs, dict, containing two dictionaries with the
        specifications about the storage systems
    '''
    
    # Specifications:
    #     soc_max: maximum state of charge (E_stor_max/Capacity)
    #     soc_min: minimum state of charge (E_stor_min/Capacity)
    #     t_cd_min: minimum time of charge/discharge ((E_stor_max - E_stor_min) /
    #                                                 P_max)
    #     eta_charge: charge efficiency (P_in/P_charge)
    #     eta_discharge: discharge efficiency (P_discharge/P_out)
    #     eta_self_discharge: self-discharge efficiency (E_(t+dt) - E_t)
                                                    
    ## Default values

    # Default battery energy storage system's (bess) specifications   
    bess_default = {'soc_max': 1,
                    'soc_min': 0.15,
                    't_cd_min': 3,
                    'eta_charge': 0.98,
                    'eta_discharge': 0.94,
                    'eta_self_discharge': 0.999,
                    }
    
    # File where to read/store the specification
    filename = 'storage_system_specs.csv'
    
    # If default flag is false, values are read from the csv file
    if not default_flag:
        try:
            storage_system_specs = pd.read_csv(filename)
            storage_system_specs.set_index('parameter', inplace=True)
            
        except:
            print('Could not open the file: {}.'.format(filename))
            print('...default values are applied.')
            default_flag = True

    # If default flag is true, default values are saved in a csv file and 
    # returned
    if default_flag:
        storage_system_specs = pd.DataFrame({'bess': bess_default,})
        # Adjusting the dataframe's format
        storage_system_specs.reset_index(inplace=True)
        storage_system_specs.rename(columns={'index':'parameter'}, \
                                     inplace=True)
        # Storing the dataframe as a csv file
        storage_system_specs.to_csv(filename, index=False)
        
        # Restoring dataframe's format
        storage_system_specs.set_index('parameter', inplace=True)
        
    # Returning the dataframe as a dictionary
    storage_system_specs = storage_system_specs.to_dict()
    
    return storage_system_specs

