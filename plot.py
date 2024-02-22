import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import sys, os
import utils


utils.SetStyle()


parser = argparse.ArgumentParser()
parser.add_argument('--plot-folder', default='plots', help='Folder to save results')
parser.add_argument('--ndata', default=600000, type=int, help='Number of data points used in the unfolding')
parser.add_argument('--data-name', default='Herwig', help='Name of the pseudo-data sample used for unfolding')
parser.add_argument('--nbins', default=30,type=int, help='Number of bins to use')
opt = parser.parse_args()

#Name of the truth distribution
truth_name = 'Truth ({})'.format('Pythia' if 'Pythia' in opt.data_name else 'Herwig')
nbins = opt.nbins
binning = [
    np.linspace(1,60,nbins),
    np.linspace(0,0.6,nbins),
    np.linspace(4,60,nbins),
    np.linspace(-14,-2,nbins),
    np.linspace(0.05,0.55,nbins),
    np.linspace(0.1,1.1,nbins),
]


if not os.path.isdir(opt.plot_folder):
    os.makedirs(opt.plot_folder)

try:
    gen = np.load("SBUnfold/clean_{}.npy".format(opt.data_name))[:opt.ndata]
    reco = np.load("SBUnfold/corrupt_{}.npy".format(opt.data_name))[:opt.ndata]
    unfolded = np.load("SBUnfold/recon_{}.npy".format(opt.data_name))[:opt.ndata]
except:
    print("ERROR: Files to load not found. Run the SBUnfold training first")

gen = utils.ReversePreprocessing(gen,'gen_features.json','JSON')
reco = utils.ReversePreprocessing(reco,'sim_features.json','JSON')
unfolded = utils.ReversePreprocessing(unfolded,'gen_features.json','JSON')


print("Loaded file with {} events".format(gen.shape[0]))

#Loading cINN training
try:
    cINN = np.load("cINN/inn_{}.npy".format(opt.data_name))[:opt.ndata]
    cINN = utils.ReversePreprocessing(cINN,'gen_features.json','JSON')
except:
    print("ERROR: Files to load not found. Run the cINN training first")

# Loading the standard diffusion training
try:
    fpcd = np.load("diffusion/diffusion_{}.npy".format(opt.data_name))[:opt.ndata]
    fpcd = utils.ReversePreprocessing(fpcd,'gen_features.json','JSON')
except:
    print("ERROR: Files to load not found. Run the Diffusion training first")

#Loading OmniFold training

try:
    if opt.ndata == 1000:
        omnifold = np.load("omnifold/omnifold_Pythia26_1000.npy")[:opt.ndata]
        omnifold = utils.ReversePreprocessing(omnifold,'gen_features.json','JSON')
        weights = np.load("omnifold/weights_data_1000.npy")[:opt.ndata]
    else:
        omnifold = np.load("omnifold/omnifold_Pythia26_1000000.npy")[:opt.ndata]
        omnifold = utils.ReversePreprocessing(omnifold,'gen_features.json','JSON')
        weights = np.load("omnifold/weights_data_1000000.npy")[:opt.ndata]
except:
    print("ERROR: Files to load not found. Run the OmniFold training first")



if 'Pythia' in opt.data_name:
    #Make the corner plots only for pythia -> pythia
    utils.overlaid_corner(
        [gen,reco,unfolded],
        ["Truth (Pythia)", "Reconstructed", "SBUnfold"],
        name = 'SBUnfold'
    )

    utils.overlaid_corner(
        [gen,reco,cINN],
        ["Truth (Pythia)", "Reconstructed", "cINN"],
        name = 'cINN'
    )


nfeatures = gen.shape[1]
feature_names = ["Jet Mass [GeV]","Jet Width", "$n_{constituents}$",r"$ln\rho$","$z_g$",r"$\tau_{21}$"]
logy = [False,True,False,False,True,False]
locs = ['best','best','best','upper left','best','upper left']
for feat in range(nfeatures):
    print("Plotting {}".format(feature_names[feat]))
    feed_dict = {
        truth_name: gen[:,feat],
        'Reconstructed': reco[:,feat],
        'cINN':cINN[:,feat],
        'SBUnfold': unfolded[:,feat],
    }

    weight_dict = {
        truth_name: np.ones(gen.shape[0]),
        'Reconstructed': np.ones(reco.shape[0]),
        'cINN':np.ones(cINN.shape[0]),
        'SBUnfold': np.ones(unfolded.shape[0]),
        }

    if 'Pythia' in opt.data_name:
        pass
        #Omnifold not needed but diffusion is loaded
        #feed_dict['FPCD'] = fpcd[:,feat]
        #weight_dict['FPCD'] = np.ones(fpcd.shape[0])
    else:
        feed_dict['OmniFold (step 1)'] = omnifold[:,feat]
        weight_dict['OmniFold (step 1)'] =weights

    
    fig,ax0 = utils.HistRoutine(feed_dict,weights = weight_dict,
                                label_loc= locs[feat],
                                xlabel=feature_names[feat], ylabel= 'Normalized entries',
                                binning = binning[feat],
                                logy=logy[feat],
                                reference_name=truth_name,
                                )
    #ax0.set_xscale("log")
    fig.savefig('{}/{}_{}_{}.pdf'.format(opt.plot_folder,opt.data_name,feat,opt.ndata))


    #make 2D plot
    fig,gs = utils.SetGrid(ratio=False)
    ax = plt.subplot(gs[0])
    cmap = plt.get_cmap('viridis')
    cmap.set_bad("white")
    h,x,y=np.histogram2d(unfolded[:,feat], reco[:,feat],  bins=(binning[feat], binning[feat]))
    h = np.ma.divide(h,np.sum(h,-1,keepdims=True)).filled(0)
    h[h==0]=np.nan
    im = ax.pcolormesh(binning[feat],binning[feat],h,cmap=cmap)
    fig.colorbar(im, ax=ax,label='Normalized Entries')
    ax.set_ylabel('Unfolded {}'.format(feature_names[feat]),fontsize=20)
    ax.set_xlabel('Reconstructed {}'.format(feature_names[feat]),fontsize=20)

    fig.savefig('{}/{}_{}_2D.pdf'.format(opt.plot_folder,opt.data_name,feat))

