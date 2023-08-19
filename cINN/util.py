'''
Implementation of functions that are important for training.
'''


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


@tf.function
def train_density_estimation(flow, optimizer, batch_gen, batch_sim):
    """
    Train function for density estimation normalizing flows.
    :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
    :param batch: Batch of the train data.
    :return: loss.
    """
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(flow.log_prob(batch_gen, context=batch_sim))  # negative log likelihood
    gradients = tape.gradient(loss, flow.trainable_weights)
    optimizer.apply_gradients(zip(gradients, flow.trainable_weights))

    return loss


def plot_loss(loss, log_dir = ".", name="", log_axis=True):
    """Plot the traings curve"""
    fig, ax1 = plt.subplots(1, figsize=(10,4))
    epoch = np.arange(len(loss))
    loss = np.array(loss)
    
    if log_axis:
        ax1.set_yscale('log')

    loss_name = name + " Loss"
    plt.plot(epoch, loss[:], color='red', markersize=12, label=r'%s' %loss_name)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=9, fancybox=True, shadow=True, prop={'size':10})
    ax1.set_xlabel(r'Epochs')
    ax1.set_ylabel(r'Loss')
    fig.savefig( log_dir + '/%s.pdf' %name, dpi=120, bbox_inches="tight")
    plt.close('all')


def plot_tau_ratio(true, gen, detector, name='tau_ratio'):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    BINS = 50
    gcolor = '#3b528b'
    dcolor = '#e41a1c'
    FONTSIZE = 16

    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios' : [3, 1], 'hspace' : 0.00})
    plt.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.18)

    ratio_true = true[:,1]/true[:,0]
    ratio_gen = gen[:,1]/gen[:,0]
    ratio_detector = detector[:,1]/detector[:,0]

    y_t, x_t = np.histogram(ratio_true, BINS, range=[0,1.2])
    y_p, x_p = np.histogram(ratio_gen, BINS, range=[0,1.2])
    y_d, x_d = np.histogram(ratio_detector, BINS, range=[0,1.2])

    line_dat, = axs[0].step(x_t[:BINS], y_t, dcolor, label='Truth', linewidth=1.0, where='mid')
    line_gen, = axs[0].step(x_p[:BINS], y_p, gcolor, label='cINN', linewidth=1.0, where='mid')
    line_det, = axs[0].step(x_d[:BINS], y_d, 'green', label='Detector', linewidth=1.0, where='mid')

    for j in range(2):
        for label in ( [axs[j].yaxis.get_offset_text()] +
                      axs[j].get_xticklabels() + axs[j].get_yticklabels()):
            label.set_fontsize(FONTSIZE-2)

    axs[0].set_ylabel(r'Normalized Cross Section', fontsize = FONTSIZE)

    axs[0].legend(
        [line_gen, line_dat, line_det],
        ['cINN', 'Truth', 'Detector'],
        #title = "GAN vs Data",
        loc='upper left',
        prop={'size':(FONTSIZE-2)},
        frameon=False)

    # lower panel

    axs[1].set_ylabel(r'$\frac{\text{cINN}}{\text{Truth}}$', fontsize = FONTSIZE)

    dummy = 1.0
    y_r = (y_p)/((y_t + dummy))

    #statistic
    axs[1].step(x_p[:BINS], y_r, 'black', linewidth=1.0, where='mid')
    axs[1].set_ylim((0.85,1.15))
    axs[1].set_yticks([0.9, 1.0, 1.1])
    axs[1].set_yticklabels([r'$0.9$', r'$1.0$', "$1.1$"])
    axs[1].axhline(y=1,linewidth=1, linestyle='--', color='grey')
    axs[1].set_xlabel(r'$N$-subjettiness ratio $\tau_{21}$', fontsize = FONTSIZE)

    fig.savefig(f'{name}.pdf', format='pdf')
    plt.close()

def plot_pull(true, gen, detector, name='tau_ratio'):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    BINS = 50
    gcolor = '#3b528b'
    dcolor = '#e41a1c'
    FONTSIZE = 16

    fig, axs = plt.subplots(1, 1)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)

    ratio_true = true[:,1]/true[:,0]
    ratio_gen = gen[:,1]/gen[:,0]
    ratio_detector = detector[:,1]/detector[:,0]

    y_t, x_t = np.histogram(ratio_detector-ratio_true, BINS, range=[-1.2,1.2])
    y_p, x_p = np.histogram(ratio_detector-ratio_gen, BINS, range=[-1.2,1.2])

    line_dat, = axs.step(x_t[:BINS], y_t, dcolor, label='Truth', linewidth=1.0, where='mid')
    line_gen, = axs.step(x_p[:BINS], y_p, gcolor, label='cINN', linewidth=1.0, where='mid')

   
    for label in ( [axs.yaxis.get_offset_text()] +
                    axs.get_xticklabels() + axs.get_yticklabels()):
        label.set_fontsize(FONTSIZE-2)

    axs.set_ylabel(r'Normalized', fontsize = FONTSIZE)

    axs.legend(
        [line_gen, line_dat],
        ['cINN', 'Truth'],
        #title = "GAN vs Data",
        loc='upper left',
        prop={'size':(FONTSIZE-2)},
        frameon=False)

    # lower panel
    axs.set_xlabel(r'Pull $N$-subjettiness ratio $\tau_{21}$', fontsize = FONTSIZE)

    fig.savefig(f'{name}.pdf', format='pdf')
    plt.close()

def plot_calibration(y_inn, y_data, name = r'\tau_{21}', n_quantiles = 100):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    y_inn_events = y_inn[:,1,:]/y_inn[:,0,:]
    y_data_events = y_data[:,1]/y_data[:,0]
    
    netcolor = '#e41a1c'
    FONTSIZE = 16

    fig, axs = plt.subplots(1, 1)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)
    #samples_sorted = np.sort(y_inn_events, axis = 1)
    quantiles = np.sum(y_inn_events<np.ones(y_inn_events.shape) *y_data_events[:, np.newaxis], axis = -1)
    calibration = np.sum(np.arange(n_quantiles) * np.ones((quantiles.shape[0], n_quantiles)) < np.ones((quantiles.shape[0], n_quantiles)) * quantiles[:, np.newaxis], axis = 0)
    calibration_x = np.arange(n_quantiles)/n_quantiles
    calibration_y = calibration/quantiles.shape[0]
    #Histogram
    axs.step(calibration_x, calibration_y, label="cINN", color=netcolor, linewidth=1.0, where='mid')
    axs.axline((0, 1), (1, 0), linewidth=1, linestyle='--', color='grey')
    axs.legend(loc='lower left', frameon=False)
    fig.savefig(f'calibration.pdf', bbox_inches='tight', format='pdf', pad_inches=0.05)
    plt.close()
