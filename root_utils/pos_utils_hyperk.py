import numpy as np

row_remap = np.flip(np.arange(75))

"""
The index starts at 1 and counts up continuously with no gaps
"""


def is_barrel(pmt_index):
    """Returns True if pmt is in the Barrel"""
    return (pmt_index < 22464) | ((pmt_index >= 29988) & (pmt_index < 30924))


def is_bottom(pmt_index):
    """Returns True if pmt is in the bottom cap"""
    return (pmt_index >= 30924) & (pmt_index < 38448)


def is_top(pmt_index):
    """Returns True if pmt is in the top cap"""
    return (pmt_index >= 22464) & (pmt_index < 29988)


def rearrange_barrel_indices(pmt_index):
    """rearrange indices to have consecutive module 
    indexing starting with top row in the barrel"""

    # check if there are non-barrel indices here
    is_not_barrel = ~is_barrel(pmt_index)
    any_not_barrel = np.bitwise_or.reduce(is_not_barrel)
    if any_not_barrel:
        raise ValueError('Passed a non-barrel PMT for geometry processing')

    # HyperK is indexed 3 rows at a time, starts with 3 rows of the first column, then moves to next column,
    # until those 3 rows are complete. Then it moves on to the next three rows. Rearrange to do 1 row at a time.
    # And the bottom three rows are done in reverse order (bottom and third from bottom are swapped).

    barrel_bulk_indices = np.where(pmt_index < 21528)
    barrel_top_3rows_indices = np.where(((pmt_index >= 29988) & (pmt_index < 30924)))
    barrel_bottom_3rows_indices = np.where(((pmt_index >= 21528) & (pmt_index < 22464)))

    three_row_index = np.zeros_like(pmt_index)
    three_row_index[barrel_bulk_indices] = (pmt_index[barrel_bulk_indices]//936) + 1
    three_row_index[barrel_top_3rows_indices] = 0
    three_row_index[barrel_bottom_3rows_indices] = 24

    row_in_3rows = np.zeros_like(pmt_index)
    row_in_3rows[barrel_bulk_indices] = pmt_index[barrel_bulk_indices] % 3
    row_in_3rows[barrel_top_3rows_indices] = (pmt_index[barrel_top_3rows_indices] - 29988) % 3
    row_in_3rows[barrel_bottom_3rows_indices] = (2-pmt_index[barrel_top_3rows_indices]) % 3

    column_index = np.zeros_like(pmt_index)
    column_index[barrel_bulk_indices] = (pmt_index[barrel_bulk_indices]//3) % 312
    column_index[barrel_top_3rows_indices] = ((pmt_index[barrel_top_3rows_indices] - 29988) // 3) % 312
    column_index[barrel_bottom_3rows_indices] = (pmt_index[barrel_bulk_indices]//3) % 312

    rearranged_module_index = 312*(3*three_row_index + row_in_3rows) + column_index

    return rearranged_module_index


def row_col_rearranged(rearranged_barrel_index):
    """return row and column index based on the rearranged module indices"""
    row = row_remap[rearranged_barrel_index//312]
    col = rearranged_barrel_index % 312
    return row, col


def row_col(pmt_index):
    """return row and column from a raw module index"""
    return row_col_rearranged(rearrange_barrel_indices(pmt_index))
