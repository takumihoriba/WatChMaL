import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.image as mpimg
import numpy as np
import torch

def plotter2(tensor_data, iteration, labels):
    plt.imshow(tensor_data[1, 1, :, :], cmap='viridis', interpolation='nearest')
    plt.colorbar(label = 'Charge')
    if labels[1] == 1:
        plt.title('Electron cherenkov ring')
    else:
        plt.title('Gamma cherenkov ring')
    plt.xlabel('Horizontal channels')
    plt.ylabel('Vertical channels')
    #plt.title(f'')
    print(f'saving figure {iteration}')
    plt.savefig(f'/fast_scratch/ipress/egamma/tensor-plots-validation/tensor_plot_{iteration}.png')
    plt.close()


def mover(tensor_data):
    #sum up the columns and rows and take their mean
    Column_sum = torch.sum(tensor_data[0, 1, :, :], axis = 0)
    Row_sum = torch.sum(tensor_data[0, 1, :, :], axis = 1)
    Column_mean = torch.mean(Column_sum)
    Row_mean = torch.mean(Row_sum)
    Column_sum = list(Column_sum)
    Row_sum = list(Row_sum)
    #take the difference between the mean and the desired channel you want to center around
    mean_diff_row = np.round(Row_mean - 80)
    mean_diff_col = np.round(Column_mean - 80)
    print('mean_diff_row = ', mean_diff_row)
    print('mean_diff_col = ', mean_diff_col)
    n_column = len(Column_sum)
    n_row = len(Row_sum)
    print('n_row', n_row)
    
    position_row = int(mean_diff_row)
    position_column = int(mean_diff_col)
    print('pos_row', position_row)
    print('pos_col', position_column)
    #Splice the beginning/end of the array onto the other end
    x_col = Column_sum[:position_column]
    Column_adjusted = Column_sum[position_column:]
    Column_adjusted.extend(x_col)
    
    x_row = Row_sum[:position_row]
    Row_adjusted = Row_sum[position_row:]
    Row_adjusted.extend(x_row)
    
    print('col adjusted mean', np.mean(Column_adjusted))
    print('row adjusted mean', np.mean(Row_adjusted))

    return Column_sum, Row_sum, Column_adjusted, Row_adjusted

def matrix_calc(tensor_data, iteration = None):
    #Column_sum = torch.sum(tensor_data[0, 1, :, :], axis = 0)
    #Row_sum = torch.sum(tensor_data[0, 1, :, :], axis = 1)
    #Column_mean = torch.mean(Column_sum)
    #Row_mean = torch.mean(Row_sum)
    #mean_diff_row = Row_mean - 80
    #mean_diff_col = Column_mean - 80
    #print('col_sum', Column_sum)
    #print('row_sum', Row_sum)
    Column_sum, Row_sum, Column_adjusted, Row_adjusted = mover(tensor_data)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.plot(Column_sum, label = 'columns')
    ax1.plot(Row_sum, label = 'rows')
    ax2.plot(Column_adjusted, label = 'columns adjusted')
    ax2.plot(Row_adjusted, label = 'rows adjusted')
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    ax1.set_xlabel('Channel')
    ax2.set_xlabel('Channel')
    ax1.set_ylabel('Sum of each channel')
    fig.savefig(f'/fast_scratch/ipress/egamma/tensor-plots-validation/matrix_calc_{iteration}.png')
    plt.close()
    #print('row_mean =', Row_mean)
    #print('Column_mean =', Column_mean)
    return Row_sum, Column_sum, Column_adjusted, Row_adjusted

def plotter_val(tensor_data, iteration, labels):
    plt.imshow(tensor_data[0, 1, :, :], cmap='viridis', interpolation='nearest')
    plt.colorbar(label = 'Charge')
    if labels[1] == 1:
        plt.title('Electron cherenkov ring')
    else:
        plt.title('Gamma cherenkov ring')
    plt.xlabel('Horizontal channels')
    plt.ylabel('Vertical channels')
    #plt.title(f'')
    print('tensor_data size = ', tensor_data.size())
    print(f'saving figure {iteration}')
    plt.savefig(f'/fast_scratch/ipress/egamma/tensor-plots-validation/tensor_plot_{iteration}.png')
    plt.close()
    matrix_calc(tensor_data, iteration)

