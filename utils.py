import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
import json, yaml
import corner
import matplotlib.lines as mlines

line_style = {
    'Truth':'dotted',
    'SBUnfold':'-',
    'cINN':'-',
    'OmniFold (step 1)':'-',
    'Reconstructed':'-'
    
}

colors = {
    'Truth':'black',
    'SBUnfold':'#7570b3',
    'cINN':'#2ca25f',
    'OmniFold (step 1)':'#ffb347',
    #'OmniFold (step 1) 1k':'#ff8547',
    'Reconstructed':'red'
}


def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    hep.style.use("CMS") 

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs


def GetEMD(ref,array,weights_arr,nboot = 100):
    from scipy.stats import wasserstein_distance
    ds = []
    for _ in range(nboot):
        #ref_boot = np.random.choice(ref,ref.shape[0])
        arr_idx = np.random.choice(range(array.shape[0]),array.shape[0])
        array_boot = array[arr_idx]
        w_boot = weights_arr[arr_idx]
        ds.append(wasserstein_distance(ref,array_boot,v_weights=w_boot))
    
    return np.mean(ds), np.std(ds)
    # mse = np.square(ref-array)/ref
    # return np.sum(mse)


class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.2f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

    # xposition = 0.9
    # yposition=1.03
    # text = 'H1'
    # WriteText(xposition,yposition,text,ax0)


def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')

def get_triangle_distance(x,y,binning):
    dist = 0
    w = binning[1:] - binning[:-1]
    for ib in range(len(x)):
        dist+=0.5*w[ib]*(x[ib] - y[ib])**2/(x[ib] + y[ib]) if x[ib] + y[ib] >0 else 0.0
    return dist*1e3

def HistRoutine(feed_dict,xlabel='',ylabel='',reference_name='Truth',logy=False,binning=None,label_loc='best',plot_ratio=True,weights=None,uncertainty=None,triangle=True):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    ref_plot = {'histtype':'stepfilled','alpha':0.2}
    other_plots = {'histtype':'step','linewidth':2}
    fig,gs = SetGrid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])

    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),30)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)

    maxy = 0    
    for ip,plot in enumerate(feed_dict.keys()):
        plot_style = ref_plot if reference_name == plot else other_plots
        if weights is not None:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=True,weights=weights[plot],**plot_style)
        else:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=True,**plot_style)

        if triangle:
            print(plot)
            d,err = GetEMD(feed_dict[reference_name],feed_dict[plot],weights[plot])
            print("EMD distance is: {}+-{}".format(d,err))
            d = get_triangle_distance(dist,reference_hist,binning)
            print("Triangular distance is: {}".format(d))
            
        if np.max(dist) > maxy:
            maxy = np.max(dist)
            
        if plot_ratio:
            if reference_name!=plot:
                ratio = np.ma.divide(dist,reference_hist).filled(0)                
                ax1.plot(xaxis,ratio,color=colors[plot],marker='+',ms=8,lw=0,markerfacecolor='none',markeredgewidth=3)
                if uncertainty is not None:
                    for ibin in range(len(binning)-1):
                        xup = binning[ibin+1]
                        xlow = binning[ibin]
                        ax1.fill_between(np.array([xlow,xup]),
                                         uncertainty[ibin],-uncertainty[ibin], alpha=0.3,color='k')    
    if logy:
        ax0.set_yscale('log')

    ax0.legend(loc=label_loc,fontsize=16,ncol=2)
    ax0.set_ylim(0,1.3*maxy)
    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 
        plt.ylabel('Ratio to Truth')
        plt.axhline(y=1.0, color='r', linestyle='-',linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([0.5,1.5])
        plt.xlabel(xlabel)
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
        
    return fig,ax0


def DataLoader(sample_name,
               N_t=1000000,N_v=600000,
               cache_dir="/global/cfs/cdirs/m3929/I2SB/",json_path='JSON'):
    import energyflow as ef
    datasets = {sample_name: ef.zjets_delphes.load(sample_name, num_data=N_t+N_v,
                                                   cache_dir=cache_dir,exclude_keys=['particles'])}
    feature_names = ['widths','mults','sdms','zgs','tau2s']
    gen_features = [datasets[sample_name]['gen_jets'][:,3]]
    sim_features = [datasets[sample_name]['sim_jets'][:,3]]

    for feature in feature_names:
        gen_features.append(datasets[sample_name]['gen_'+feature])
        sim_features.append(datasets[sample_name]['sim_'+feature])

    gen_features = np.stack(gen_features,-1)
    sim_features = np.stack(sim_features,-1)
    #ln rho
    gen_features[:,3] = 2*np.ma.log(np.ma.divide(gen_features[:,3],datasets[sample_name]['gen_jets'][:,0]).filled(0)).filled(0)
    sim_features[:,3] = 2*np.ma.log(np.ma.divide(sim_features[:,3],datasets[sample_name]['sim_jets'][:,0]).filled(0)).filled(0)
    #tau2
    gen_features[:,5] = gen_features[:,5]/(10**-50 + gen_features[:,1])
    sim_features[:,5] = sim_features[:,5]/(10**-50 + sim_features[:,1])

    #Standardize
    gen_features = ApplyPreprocessing(gen_features,'gen_features.json',json_path)
    sim_features = ApplyPreprocessing(sim_features,'sim_features.json',json_path)

    train_gen = gen_features[:N_t]
    train_sim = sim_features[:N_t]
    
    test_gen = gen_features[N_t:]
    test_sim = sim_features[N_t:]

    return train_gen, train_sim, test_gen,test_sim
    
def CalcPreprocessing(data,fname,base_folder):
    '''Apply data preprocessing'''
    
    data_dict = {}
    mean = np.average(data,axis=0)
    std = np.std(data,axis=0)
    data_dict['mean']=mean.tolist()
    data_dict['std']=std.tolist()
    data_dict['min']=np.min(data,0).tolist()
    data_dict['max']=np.max(data,0).tolist()    
    SaveJson(fname,data_dict,base_folder)



def ApplyPreprocessing(data,fname,base_folder):
    #CalcPreprocessing(data,fname,base_folder)    
    data_dict = LoadJson(fname,base_folder)
    data = (np.ma.divide((data-data_dict['mean']),data_dict['std']).filled(0)).astype(np.float32)
    #data = (np.ma.divide((data-data_dict['min']),np.array(data_dict['max']) - data_dict['min']).filled(0)).astype(np.float32)
    return data


def ReversePreprocessing(data,fname,base_folder):
    data_dict = LoadJson(fname,base_folder)
    #data = (np.array(data_dict['max']) - data_dict['min']) * data + data_dict['min']
    data = data * data_dict['std'] + data_dict['mean']
    data[:,2] = np.round(data[:,2]) #particle multiplicity should be an integer
    return data




def SaveJson(save_file,data,base_folder='JSON'):
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)
    
    with open(os.path.join(base_folder,save_file),'w') as f:
        json.dump(data, f)

    
def LoadJson(file_name,base_folder='JSON'):
    import json,yaml
    JSONPATH = os.path.join(base_folder,file_name)
    return yaml.safe_load(open(JSONPATH))


def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)

def overlaid_corner(samples_list, sample_labels,name=''):
    """Plots multiple corners on top of each other"""

    CORNER_KWARGS = dict(
        smooth=0.9,
        label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16),
        #quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        #show_titles=True,
        max_n_ticks=3
    )

    
    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    max_len = max([len(s) for s in samples_list])
    cmap = plt.cm.get_cmap('gist_rainbow', n)
    colors = ["black","red","blue"]

    plot_range = []
    for dim in range(ndim):
        plot_range.append(
            [
                min([min(samples_list[i].T[dim]) for i in range(n)]),
                max([max(samples_list[i].T[dim]) for i in range(n)]),
            ]
        )
    plot_range = [[3,70],[0.,0.6],[1.0,70],[-13,-3],[0,0.5],[0.1,1.2]]

    CORNER_KWARGS.update(range=plot_range)

    fig = corner.corner(
        samples_list[0],
        color=colors[0],
        labels = ["Jet Mass [GeV]","Jet Width", "$n_{constituents}$",r"$ln\rho$","$z_g$",r"$\tau_{21}$"],
        **CORNER_KWARGS
    )

    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=get_normalisation_weight(len(samples_list[idx]), max_len),
            color=colors[idx],
            **CORNER_KWARGS
        )

    plt.legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=sample_labels[i])
            for i in range(n)
        ],
        fontsize=20, frameon=False,
        bbox_to_anchor=(1, ndim), loc="upper right"
    )
    plt.savefig("plots/corner_{}.pdf".format(name))
    #plt.close()
