import numpy as np

row_remap=np.flip(np.arange(75))

"""
The index starts at 1 and counts up continuously with no gaps
"""

def is_barrel(pmt_index):
    """Returns True if pmt is in the Barrel"""
    return ((pmt_index < 22464) | ((pmt_index >= 29988) & (pmt_index < 30924)))

def is_bottom(pmt_index):
    """Returns True if pmt is in the bottom cap"""
    return ((pmt_index >= 30924) & (pmt_index < 38448))

def is_top(pmt_index):
    """Returns True if pmt is in the top cap"""
    return ((pmt_index >= 22464) & (pmt_index < 29988))

def rearrange_barrel_indices(pmt_index):
    """rearrange indices to have consecutive module 
    indexing starting with top row in the barrel"""

    #check if there are non-barrel indices here
    is_not_barrel= ~is_barrel(pmt_index)
    any_not_barrel=np.bitwise_or.reduce(is_not_barrel)
    if any_not_barrel:
        raise ValueError('Passed a non-barrel PMT for geometry processing')
    
    rearranged_module_index=np.zeros_like(pmt_index)
    barrel_bulk_indices=np.where(pmt_index < 22464)
    barrel_top_row_indices=np.where(((pmt_index >= 29988) & (pmt_index < 30924)))
    rearranged_module_index[barrel_bulk_indices]= pmt_index[barrel_bulk_indices] + 936
    rearranged_module_index[barrel_top_row_indices]= pmt_index[barrel_top_row_indices] - 29988
    return rearranged_module_index


def row_col_rearranged(rearranged_barrel_index):
    """return row and column index based on the rearranged module indices"""
    row=row_remap[rearranged_barrel_index//312]
    col=rearranged_barrel_index%312
    return row, col

def row_col(module_index):
    """return row and column from a raw module index"""
    return row_col_rearranged(rearrange_barrel_indices(module_index))

