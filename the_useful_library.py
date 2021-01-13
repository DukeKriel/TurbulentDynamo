#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import seaborn as sns


##################################################################
## FUNCTIONS
##################################################################
def str2bool(v):
    """ str2bool
    BASED ON: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "y", "1"): return True
    elif v.lower() in ("no", "false", "f", "n", "0"): return False
    else: raise argparse.ArgumentTypeError("Boolean value expected.")


def makeFilter(str_contains=None, str_not_contains=None, str_startswith=None, str_endswith=None, file_index_placing=None, file_start_index=0, file_end_index=np.inf):
    """ makeFilter
    PURPOSE: Create a filter condition for files that look a particular way.
    """
    def meetsCondition(element):
        ## if str_contains specified, then look for condition
        if str_contains is not None: bool_contains = element.__contains__(str_contains)
        else: bool_contains = True # don't consider condition
        ## if str_not_contains specified, then look for condition
        if str_not_contains is not None: bool_not_contains = not(element.__contains__(str_not_contains))
        else: bool_not_contains = True # don't consider condition
        ## if str_startswith specified, then look for condition
        if str_startswith is not None: bool_startswith = element.startswith(str_startswith)
        else: bool_startswith = True # don't consider condition
        ## if str_endswith specified, then look for condition
        if str_endswith is not None: bool_endswith = element.endswith(str_endswith)
        else: bool_endswith = True # don't consider condition
        ## make sure that the file has the right structure (i.e. there are the correct number of spacers)
        if bool_contains and bool_not_contains and bool_startswith and bool_endswith and ( file_index_placing is not None ):
            ## make sure that simulation file is in time range
            if len(element.split("_")) > abs(file_index_placing): # at least the required number of spacers
                bool_time_after  = ( int(element.split("_")[file_index_placing]) >= file_start_index )
                bool_time_before = ( int(element.split("_")[file_index_placing]) <= file_end_index )
                ## if the file meets the required conditions
                if (bool_time_after and bool_time_before): return True
        ## otherwise don"t look at the file
        else: return False
    return meetsCondition


def createFolder(folder_name):
    """ createFolder
    PURPOSE: Create a folder if and only if it does not already exist.
    """
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
        print("SUCCESS: \n\tFolder created. \n\t" + folder_name + "\n")
    else: print("WARNING: \n\tFolder already exists (folder not created). \n\t" + folder_name + "\n")


def createFilePath(folder_names):
    """ creatFilePath
    PURPOSE: Concatinate a list of folder names into a filepath string.
    """
    return ("/".join([folder for folder in folder_names if folder != ""])).replace("//", "/")


def getFilesFromFolder(folder_directory, str_contains=None, str_startswith=None, str_endswith=None, str_not_contains=None, file_index_placing=None, file_start_index=0, file_end_index=np.inf):
    ''' getFilesFromFolder
    PURPOSE: Return the names of files that meet the required conditions in the specified folder.
    '''
    myFilter = makeFilter(str_contains, str_not_contains, str_startswith, str_endswith, file_index_placing, file_start_index, file_end_index)
    return list(filter(myFilter, sorted(os.listdir(folder_directory))))


def normaliseData(vals):
    ''' normaliseData
    PURPOSE:
        Normalise values by translating and scaling the distribution of points that lie on [a,b] 
        to one with the same shape but instead lies on [0,1]. All points become scaled by: 
            (x-a)/(b-a) for all x in vals.
        (This is different to normalising points by scaling them by the largest point's magnitude from 0).
    '''
    if not(type(vals) == np.ndarray): vals = np.array(vals)
    vals = vals - vals.min()
    if (vals.max() == 0): return vals
    return vals / vals.max()


## END OF LIBRARY