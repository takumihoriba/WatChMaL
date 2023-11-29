import h5py
import numpy as np
import hashlib

import matplotlib
from matplotlib import pyplot as plt
import analysis.utils.binning as bins

def read_fitqun_file(file_path, regression=False):
    with h5py.File(file_path,mode='r') as h5fw:
        print(h5fw.keys())
        fitqun_hash = get_rootfile_eventid_hash(np.ravel(h5fw['root_files']), np.ravel(h5fw['event_ids']), fitqun=True)
        e_1rnll = np.ravel(h5fw['e_1rnll'])
        mu_1rnll = np.ravel(h5fw['mu_1rnll'])
        e_1rmom = np.ravel(h5fw['e_1rmom'])
        mu_1rmom = np.ravel(h5fw['mu_1rmom'])
        labels = np.ravel(h5fw['labels'])
        fitqun_1rmom = np.ones(len(e_1rmom))
        fitqun_1rmom[labels==0] = mu_1rmom[labels==0]
        fitqun_1rmom[labels==1] = e_1rmom[labels==1]
        discr = -e_1rnll + mu_1rnll 
        discr = discr > e_1rmom*0.2
        discr = discr.astype(int)
        temp = np.abs(labels-discr)
        print(f"fitqun e- avg: {1-np.sum(temp[labels==1])/len(temp[labels==1])}")
        print(f"fitqun mu- avg: {1-np.sum(temp[labels==0])/len(temp[labels==0])}")
        plt.hist(fitqun_1rmom, label = 'fiTQun', range=[0,1000], bins=10)
        '''
        for mom in range(0,1000,100):
            temp_discr = discr[(fitqun_1rmom > mom) & (fitqun_1rmom < mom+100) ]
            temp_labels = labels[(fitqun_1rmom > mom) & (fitqun_1rmom < mom+100) ]
            temp_error = np.abs(temp_labels-temp_discr)
            print(f"mom: {mom}, fitqun avg: {1-np.sum(temp_error)/len(temp_error)}")
        plt.xlabel("Momentum")
        plt.ylabel("Counts")
        #plt.ylim(90,100)
        #plt.figure(e_mom_fig_fitqun.number)
        plt.savefig(file_path + 'fitqun_reco_mom.png', format='png')
        '''
        if regression:
            mu_1rpos = np.array(h5fw['mu_1rpos'])
            e_1rpos = np.array(h5fw['e_1rpos'])
            print(e_1rpos.shape, mu_1rpos.shape)
            return (discr, labels, fitqun_1rmom, fitqun_hash), (mu_1rpos, e_1rpos)
        else:
            return discr, labels, fitqun_1rmom, fitqun_hash

def make_fitqunlike_discr(softmax, energies, labels):
    discr = softmax[:,1]-softmax[:,0]
    min = np.argmin(discr)
    max = np.argmax(discr)

    plt.hist2d(energies, softmax[:,1]-softmax[:,0], norm=matplotlib.colors.LogNorm(), cmap=matplotlib.cm.gray)
    plt.colorbar()
    plt.savefig("outputs/2d_softmax_pt_hist.png")
    
    for scale in np.logspace(-4,0,1000):
        temp_discr = discr > energies*scale
        temp_discr = temp_discr.astype(int)
        temp_calc = np.abs(labels-discr)
        temp_metric = np.sum(temp_calc)/len(temp_discr)

def get_rootfile_eventid_hash(rootfiles, event_ids, fitqun=True):
    """Converts rootfile + event id into 1 unique ID, to link fitqun + ml output

    Args:
        rootfiles (_type_): array of event rootfiles from h5 file
        event_ids (_type_): array of event ids from h5 file
        fitqun (bool, optional): If this is for the fitqun file. Defaults to True.

    Returns:
        list of unique IDs: should be unique ID for each rootfile+event id combination
    """
    print(event_ids)
    #We skip the 0th event in skdetsim root->h5 because (???)
    #This is to correct for that
    if fitqun==False:
        event_ids = np.array([x+1 for x in event_ids])
    print(event_ids)
    string_eventids = np.char.mod('%d', event_ids)
    string_rootfiles = [x.decode('UTF-8') for x in rootfiles]
    string_rootfiles = np.char.split(string_rootfiles,'/')
    string_rootfiles = [x[-1] for x in string_rootfiles]
    #string_rootfiles = [x.split('/')[-1] for x in string_rootfiles]
    if fitqun:
        string_rootfiles = [x.split('_')[1] for x in string_rootfiles]
        string_rootfiles = [x.split('.')[0] for x in string_rootfiles]
    else:
        string_rootfiles = [x.split('_')[-1] for x in string_rootfiles]
        string_rootfiles = [x.split('.')[0] for x in string_rootfiles]
    combined_hash = np.char.add(string_eventids, string_rootfiles)
    combined_hash = [int(x) for x in combined_hash]
    '''
    print(f'string rootfiles:{combined_hash}')
    for i, item in enumerate (combined_hash):
        m = hashlib.md5()
        m.update(combined_hash[i].encode('UTF-8'))
        combined_hash[i] = int(str(int(m.hexdigest(), 16))[0:12])
    '''
    return combined_hash

def plot_fitqun_comparison(plot_output, ax_e, ax_fitqun_e, ax_mu, ax_fitqun_mu, name, x_axis_name, print_out_acc=False):
        #line 0 is the data, line 1 is the lower error, line 2 higher error
        ve_xdata = ax_e.lines[0].get_xdata()
        e_ml = ax_e.lines[0].get_ydata()
        e_fitqun = ax_fitqun_e.lines[0].get_ydata()
        mu_ml = ax_mu.lines[0].get_ydata()
        mu_fitqun = ax_fitqun_mu.lines[0].get_ydata()
        if print_out_acc:
            print(f'x: {ve_xdata}')
            print(f'ml e %: {e_ml}')
            print(f'fq e %: {e_fitqun}')
            print(f'ml mu mis-%: {mu_ml}')
            print(f'fq mu mis-%: {mu_fitqun}')
        e_ml_low_err=None
        e_ml_hi_err=None
        mu_ml_low_err=None
        mu_ml_hi_err=None
        e_fitqun_low_err=None
        e_fitqun_hi_err=None
        mu_fitqun_low_err=None 
        mu_fitqun_hi_err=None 
        if len(ax_e.lines) == 3:
            e_ml_low_err = ax_e.lines[1].get_ydata()
            e_ml_hi_err = ax_e.lines[2].get_ydata()
        if len(ax_mu.lines) == 3:
            mu_ml_low_err = ax_mu.lines[1].get_ydata()
            mu_ml_hi_err = ax_mu.lines[2].get_ydata()
        if len(ax_fitqun_e.lines) == 3:
            e_fitqun_low_err = ax_fitqun_e.lines[1].get_ydata()
            e_fitqun_hi_err = ax_fitqun_e.lines[2].get_ydata()
        if len(ax_fitqun_mu.lines) == 3:
            mu_fitqun_low_err = ax_fitqun_mu.lines[1].get_ydata()
            mu_fitqun_hi_err = ax_fitqun_mu.lines[2].get_ydata()


        plt.errorbar(ve_xdata, e_ml, yerr=[e_ml-e_ml_low_err, e_ml_hi_err-e_ml], color='blue', label = 'ML', linestyle='', capsize=3)
        plt.errorbar(ve_xdata, e_fitqun, yerr=[e_fitqun-e_fitqun_low_err, e_fitqun_hi_err-e_fitqun], color='red', label = 'fiTQun', linestyle='', capsize=3)
        plt.xlabel(x_axis_name)
        plt.ylabel("Electron Tagging Efficiency [%]")
        plt.legend()
        #plt.ylim(90,100)
        #plt.figure(e_mom_fig_fitqun.number)
        plt.savefig(plot_output + 'e_'+name+'.png', format='png')
        plt.clf()
        plt.errorbar(ve_xdata, mu_ml, yerr=[mu_ml-mu_ml_low_err, mu_ml_hi_err-mu_ml], color='blue', label = 'ML', linestyle='', capsize=3)
        plt.errorbar(ve_xdata, mu_fitqun, yerr=[mu_fitqun-mu_fitqun_low_err, mu_fitqun_hi_err-mu_fitqun], color='red', label = 'fiTQun', linestyle='', capsize=3)
        plt.xlabel(x_axis_name)
        plt.ylabel("Muon Mis-Tagging Efficiency [%]")
        plt.legend()
        #plt.ylim(90,100)
        #plt.figure(e_mom_fig_fitqun.number)
        plt.savefig(plot_output + 'mu_'+name+'.png', format='png')
        plt.clf()



def regression_analysis(true_positions, pred_positions):
    '''
    - scatter plot of pred vs true
    - residual histogram
    '''
    vertex_axis = ['X', 'Y', 'Z']
    for i in range(3): 
        line = np.linspace(-1600,1600,10)
        plt.figure(figsize=(5,5))
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
        plt.scatter(true_positions[:,i], pred_positions[:,i], alpha=0.05, s=0.1, color=color)
        plt.plot(line, line, '--', color='black', alpha=0.5)
        plt.xlim(-2000,2000)
        plt.ylim(-2000,2000)
        
        plt.title(f'Event Vertex for {vertex_axis[i]} Axis')
        plt.xlabel('True Position [cm]')
        plt.ylabel('Predicted Position [cm]')
        plt.show()

        residuals = true_positions[:,i] - pred_positions[:,i]
        residuals_div = residuals/(true_positions[:,i])
        residuals_cut = [] 
        residuals_div_cut = []
        cut = 1600
        
        for r in range(len(residuals)):
            if -cut < residuals[r] <  cut:
                residuals_cut.append(residuals[r])
            if -cut < residuals_div[r] < cut:
                residuals_div_cut.append(residuals_div[r])

        plt.hist(residuals_cut, bins=50, alpha=0.7, color=color)
        plt.xlim(-cut, cut)

        plt.title(f'Event Vertex for {vertex_axis[i]} Axis')
        plt.xlabel('true - predicted [cm]')
        plt.ylabel('count')
        plt.show()

        plt.hist(residuals_div_cut, bins=50, alpha=0.7, color=color)
        plt.xlim(-cut, cut)
        

        plt.title(f'Event Vertex for {vertex_axis[i]} Axis')
        plt.xlabel('(true - predicted)/true')
        plt.ylabel('count')
        plt.show()
        

def un_normalize(data, x_bounds=(-1600,1600), y_bounds=(-1600,1600), z_bounds=(-1600,1600)):
    '''
    '''
    bounds = [x_bounds, y_bounds, z_bounds]
    for i in range(3):
        data[i] = ((data[i])*(bounds[i][1] - bounds[i][0])) + bounds[i][0] 
        
    return data

def plot_regression_results(file_path = '/fast_scratch_2/aferreira/t2k/ml/data/oct20_combine_flatE/fitqun_combine.hy'):
    '''
    looking at 
    e_1rpos[3]  	        	Electron hypothesis vertex position (3D, x=0, y=1, z=2)
    mu_1rpos[3] 	                Muon hypothesis vertex position (3D, x=0, y=1, z=2)
    '''
    (discr, labels, fitqun_1rmom, fitqun_hash), (mu_1rpos, e_1rpos) = read_fitqun_file(file_path, regression=True)

    dirpath = '/fast_scratch_2/aferreira/t2k/ml/data/oct20_combine_flatE/nov26_normed_regression_only/20092023-101855/'
    true_positions = np.load(dirpath + "true_positions.npy")
    tp, = []
    for t in true_positions:
        tp.append(un_normalize(t))
    true_positions = np.array(tp)

    # in future read in same generic plotting stuff
    regression_analysis(true_positions, e_1rpos)
    regression_analysis(true_positions, mu_1rpos)

    