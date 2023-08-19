import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import sys, os
import utils


utils.SetStyle()


nbins = 20
binning = [
    np.linspace(1,60,nbins),
    np.linspace(0,0.6,nbins),
    np.linspace(1,70,nbins),
    np.linspace(-14,-2,nbins),
    np.linspace(0,0.5,nbins),
    np.linspace(0.1,1.2,nbins),
]


parser = argparse.ArgumentParser()
#parser.add_argument('--niter', type=int,default=60000, help='Iteration to load the trained model from SB')
parser.add_argument('--plot-folder', default='plots', help='Folder to save results')
parser.add_argument('--ndata', default=600000, type=int, help='Number of data points used in the unfolding')
parser.add_argument('--data-name', default='Herwig', help='Name of the sample to unfold')
opt = parser.parse_args()

if not os.path.isdir(opt.plot_folder):
    os.makedirs(opt.plot_folder)


gen = np.load("SB/clean_{}.npy".format(opt.data_name))[:opt.ndata]
reco = np.load("SB/corrupt_{}.npy".format(opt.data_name))[:opt.ndata]
unfolded = np.load("SB/recon_{}.npy".format(opt.data_name))[:opt.ndata]

gen = utils.ReversePreprocessing(gen,'gen_features.json','JSON')
reco = utils.ReversePreprocessing(reco,'sim_features.json','JSON')
unfolded = utils.ReversePreprocessing(unfolded,'gen_features.json','JSON')


print("Loaded file with {} events".format(gen.shape[0]))

#Loading cINN training

cINN = np.load("cINN/inn_{}.npy".format(opt.data_name))[:opt.ndata]
cINN = utils.ReversePreprocessing(cINN,'gen_features.json','JSON')


#Loading OmniFold training

if opt.ndata == 1000:
    omnifold = np.load("omnifold/omnifold_Pythia26_1000.npy")[:opt.ndata]
    omnifold = utils.ReversePreprocessing(omnifold,'gen_features.json','JSON')
    weights = np.load("omnifold/weights_data_1000.npy")[:opt.ndata]
else:
    omnifold = np.load("omnifold/omnifold_Pythia26_1000000.npy")[:opt.ndata]
    omnifold = utils.ReversePreprocessing(omnifold,'gen_features.json','JSON')
    weights = np.load("omnifold/weights_data_1000000.npy")[:opt.ndata]



if 'Pythia' in opt.data_name:
    #Make the corner plots only for pythia -> pythia
    utils.overlaid_corner(
        [gen,reco,unfolded],
        ["Truth", "Reconstructed", "SBUnfold"],
        name = 'SBUnfold'
    )

    utils.overlaid_corner(
        [gen,reco,cINN],
        ["Truth", "Reconstructed", "cINN"],
        name = 'cINN'
    )


nfeatures = gen.shape[1]
feature_names = ["Jet Mass [GeV]","Jet Width", "$n_{constituents}$",r"$ln\rho$","$z_g$",r"$\tau_{21}$"]
locs = ['best','best','best','upper left','best','upper left']
for feat in range(nfeatures):
    print(feat)
    feed_dict = {
        'Truth': gen[:,feat],
        'Reconstructed': reco[:,feat],
        'cINN':cINN[:,feat],
        'SBUnfold': unfolded[:,feat],
        #'OmniFold (step 1)':omnifold[:,feat],
        #'OmniFold (step 1) 1k':omnifold_1k[:,feat],
    }

    weight_dict = {
        'Truth': np.ones(gen.shape[0]),
        'Reconstructed': np.ones(reco.shape[0]),
        'cINN':np.ones(cINN.shape[0]),
        'SBUnfold': np.ones(unfolded.shape[0]),
        }
    
    fig,ax0 = utils.HistRoutine(feed_dict,weights = weight_dict,
                                label_loc= locs[feat],
                                xlabel=feature_names[feat], ylabel= 'Normalized entries',
                                binning = binning[feat],
                                logy=False,)
    #ax0.set_xscale("log")
    fig.savefig('{}/{}_{}_{}.pdf'.format(opt.plot_folder,opt.data_name,feat,opt.ndata))

