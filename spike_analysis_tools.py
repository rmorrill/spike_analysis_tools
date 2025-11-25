import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat 
from scipy.signal import savgol_filter
import pprint as pp
import matplotlib.gridspec as gridspec
import pandas as pd
from rapidfuzz import fuzz
from scipy import stats 
from kilosort.data_tools import (
    mean_waveform, cluster_templates, get_good_cluster, get_cluster_spikes,
    get_spike_waveforms, get_best_channel
)
from kilosort.io import load_ops, get_total_samples, bfile_from_ops
from pathlib import Path
import re
import pdb
from time import perf_counter as getsecs 



def plot_channel(ch, rawdata, t1, t2, Fs):
    """
    Plot raw traces 

    Parameters:
    ch (int)
    rawdata - output of makeMemMapRaw in DemoReadSGLXData.readSGLX
    t1 (float)
    t2 (float)
    Fs (float)

    Returns:
    currData (numpy array)
    
    """
    firstSamp = int(t1*Fs)
    lastSamp =  int(t2*Fs)
    currData = rawdata[ch, firstSamp:lastSamp]
    currData = np.squeeze(currData)
    t_plot = (np.array(range(firstSamp, lastSamp))+1)/Fs
    plt.figure()
    plt.plot(t_plot, currData)
    plt.title(f'channel {ch}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amp')
    currData = np.array(currData)
    return currData

def get_trig_times(signal, t_signal, verbose_flag=True): 
    signal = np.squeeze(signal)
    signal_max = np.max(signal)
    signal_min = np.min(signal)

    # let's start with the simple thing - diff of max and min 
    thresh = np.abs(signal_max - signal_min)/2 + signal_min

    if verbose_flag: 
        print(f'signal max: {signal_max}; signal min: {signal_min}')
        print(f'thresh: {thresh}')

    thresh_signal = np.where(signal > thresh, 1, 0)
    thresh_diff = np.diff(thresh_signal)
    go_up = np.where(thresh_diff == 1)[0]
    go_down = np.where(thresh_diff == -1)[0]
    trig_on = t_signal[go_up]
    trig_off = t_signal[go_down]

    if verbose_flag: 
        print(f'found {len(go_up)} trigger ups and {len(go_down)} trigger downs')

    return trig_on, trig_off, go_up, go_down

def return_spike_array(spike_train, trig_times, time_win, trial_filt=None): 
    """
    Get spike array

    Parameters:
    spike_train (numpy ndarray)
    trig_times
    time_win
    trial_filt

    Returns:
    spike_array (list)
    
    """
    
    spike_train = np.array(spike_train)

    if trial_filt is not None: 
        trig_times = trig_times[trial_filt]

    n_trs = len(trig_times)

    print(f'returning spikes for {n_trs} trs')

    spike_array = []
    for curr_trig in trig_times: 
        # find spikes in window
        twin_start = curr_trig + time_win[0]
        twin_end = curr_trig + time_win[1]
        curr_tr_spikes = spike_train[((spike_train>=twin_start) & (spike_train<twin_end))]-curr_trig
        spike_array.append(np.array(curr_tr_spikes))
    
    return spike_array
        

def return_PSTH(spike_array, t1, t2, binwidth):
    n_trs = len(spike_array) 
    psth_bins = np.arange(t1, t2 + binwidth, binwidth)
    
    spike_hist_all = np.array([np.histogram(spikes_trial, psth_bins)[0] for spikes_trial in spike_array])
    spike_hist_all = spike_hist_all/binwidth
    psth_mu = np.mean(spike_hist_all, axis=0)
    psth_std = np.std(spike_hist_all, axis=0)
    psth_sem = psth_std / np.sqrt(n_trs)
    time = np.diff(psth_bins)/2 + psth_bins[:-1]

    return psth_mu, psth_std, psth_sem, time


def plot_PSTH(psth_mu, psth_err, psth_t, col=None, ax=None):
    if col is None: 
        col = 'b'
    if ax is not None: 
        plt.sca(ax)
    else: 
        ax = plt.gca()
    linehand, = plt.plot(psth_t, psth_mu, color=col)
    plt.fill_between(psth_t, psth_mu-psth_err, psth_mu+psth_err, color=col, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('FR (Hz)')
    ax.margins(x=0)
    box_off(ax)
    return ax, linehand


def plot_raster(spike_array, t1, t2, col=None, ax=None, xlabel_off=False): 
    if col is None: 
        col = 'k'
    if ax is not None: 
        plt.sca(ax)

    plthands =  plt.eventplot(spike_array, colors=col, linewidths=1.5)
    ax = plt.gca()
    ax.margins(x=0,y=0)
    ax.set_xlim(t1, t2)
    ax.set_ylim(0,len(spike_array))
    box_off(ax)
    plt.ylabel('Trials')
    if not(xlabel_off): 
        plt.xlabel('Time (s)')
    else: 
        ax.get_xaxis().set_ticks([])
    return ax 


def box_off(ax): 
    if not(isinstance(ax, list)): 
        if isinstance(ax, np.ndarray):
            ax_list = ax.ravel()
        else: 
            ax_list = [ax]
    else: 
        ax_list = ax.copy()

    for ax in ax_list: 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

#def scale_rasters_trial_count(axs): 
def create_binary_vector(strings_list, key, use_regex=False):
    """
    Create a [0, 1] binary vector for trial filtering from a string list.

    Parameters:
    strings_list (list): list of e.g. stimuli
    key (str): what to look for

    Returns:
    
    """
    return match_strings(key, strings_list, use_regex=use_regex)


def match_strings(key, strings_list, use_regex=False):
    if use_regex:
        pattern = re.compile(key, re.IGNORECASE)
        return [1 if pattern.search(s) else 0 for s in strings_list]
    else:
        key_lower = key.lower()
        return [1 if key_lower in s.lower() else 0 for s in strings_list]

def convert_list_to_bool(vec): 
    vec_array = np.array(vec)
    return vec_array.astype(bool)

def plot_stim_lines(ax, t=0, t_off=None): 
    yL = ax.get_ylim()
    ax.plot([t, t], yL, 'k--')
    if t_off is not None: 
        ax.plot([t_off, t_off], yL, 'k--')
    return


def read_mat(matfile_full, struct, field=None):
    m = loadmat(matfile_full)
    print(f'loaded matfile: {matfile_full}')
    mkeys = [k for k in m.keys()]
    #print('found the following variables')
    #pp.pprint(mkeys)
    s = m[struct]
    vals = s[0,0]
    k_tmp = s[0,0].dtype.descr
    keys = [k[0] for k in k_tmp]
    #pdb.set_trace()
    if field is None: 
        # return whole dictionary
        struct_dict = {}
        for ii, key in enumerate(keys):
            #print(key)
            val = process_mat_data(vals[key])
            exec(key + '=val')
            exec(f"struct_dict['{key}'] = {key}")
        
        return struct_dict
    elif field in keys:
        #print(field)
        val = process_mat_data(vals[field])
        #if vals[field].size>0:
        #    val = np.squeeze(vals[field][0])
        #else: 
        #    val = []
        return val

    else: 
        print(f'ERROR: {field} not found in struct {struct}')
        return


def process_mat_data(data_in): 

    if data_in.size>0:
        if 1 in data_in.shape: 
            data_out = np.squeeze(data_in[0])
        else:
            data_out = data_in 
        # deal with string lists to convert to list
        if data_out.shape != (): 
            if type(data_out[0]) is np.ndarray:                 
                nonempty_idx = first_nonempty(data_out)
                if nonempty_idx is None: 
                    nonempty_idx = 0
                if type(data_out[nonempty_idx][0]) is np.str_: 
                    data_out = [str(d[0]) for d in data_out]
    else: 
        data_out = []

    return data_out



def first_nonempty(obj_arr):
    """
    Return (index, array) for the first non-empty element
    in a 1-D NumPy object array of arrays.  
    Raises ValueError if every element is empty.
    """
    #if obj_arr.dtype != object:
    #    raise TypeError("Input must be a 1-D NumPy object array.")

    try:
        idx = next(i for i, a in enumerate(obj_arr) if a.size)   # a.size == 0 for empty arrays
    except StopIteration:
        #pdb.set_trace()
        idx = None
        #raise ValueError("All sub-arrays are empty.")

    return idx
        

def plot_raster_psth_stack(gs_subplot, fighand, spike_array, t1, t2, binwidth, rast_col=None, psth_col=None): 
    gs_subd = gs_subplot.subgridspec(2, 1)
    ax_rast = fighand.add_subplot(gs_subd[0])
    plot_raster(spike_array, t1, t2, rast_col, ax_rast, xlabel_off=True)

    psth_mu, _, psth_sem, psth_t = return_PSTH(spike_array, t1, t2, binwidth)
    psth_mu_sm = smooth_signal(psth_mu)
    psth_sem_sm = smooth_signal(psth_sem)

    ax_psth = fighand.add_subplot(gs_subd[1])
    plot_PSTH(psth_mu_sm, psth_sem_sm, psth_t, psth_col, ax_psth)
    return ax_rast, ax_psth


def smooth_signal(data, window_length=5): 
    smoothed_data = savgol_filter(data, window_length=window_length, polyorder=2)
    return smoothed_data 

###################
# these are going to be the tools that return data easily 

def get_spikes(spikesort_path: str | Path, which_clusts=None, label: str = 'good', spike_times_sec=True): 
    """
    Return spikes 
    OLD version: get_spikes(data_path, spikesort_dir, which_clusts=None, label='good')
    """
    #spikesort_path = Path(data_path) / spikesort_dir
    spikesort_path = Path(spikesort_path)
    spike_time_file = spikesort_path / 'spike_times.npy'
    spike_cluster_file = spikesort_path / 'spike_clusters.npy'
    cluster_group_file = spikesort_path / 'cluster_group.tsv'
    cluster_KSlabel_file = spikesort_path / 'cluster_KSlabel.tsv'
    if not(cluster_KSlabel_file.exists()) : 
        cluster_KSlabel_file = spikesort_path / 'cluster_KSLabel.tsv'
    ops_file = spikesort_path / 'ops.npy'
    ops = np.load(ops_file, allow_pickle=True)
    ops = ops.item()
    
    spike_times = np.load(spike_time_file)    
    if spike_times_sec: 
        spike_times = spike_times/ops['fs']
    spike_clusters = np.load(spike_cluster_file)
    cluster_group = pd.read_csv(cluster_group_file, sep='\t')
    cluster_group_KS = pd.read_csv(cluster_KSlabel_file, sep='\t')

    # implement filters 
    if (which_clusts is None) and (label != 'all'): 
        print(f'get_spikes: no unit filter provided, returning label filter for  {label} clusts')
        which_clusts = cluster_group_KS[np.isin(cluster_group_KS['KSLabel'], label)]['cluster_id']
        #which_clusts = cluster_group[np.isin(cluster_group['KSLabel'], label)]['cluster_id']
        spike_filter = np.isin(spike_clusters, which_clusts)
    elif label == 'all': 
        print(f'get_spikes: returning label filter for all clusts')
        spike_filter = np.ones_like(spike_times).astype(bool)
        which_clusts = np.unique(spike_clusters)
    else: 
        print(f'get_spikes: returning spikes in clusts {which_clusts}')
        spike_filter = np.isin(spike_clusters, which_clusts)

    spike_times = spike_times[spike_filter]
    spike_clusters = spike_clusters[spike_filter]
    cluster_group = cluster_group_KS[np.isin(cluster_group_KS['cluster_id'], which_clusts)]


    print(f'found {len(spike_times)} spikes')

    return spike_times, spike_clusters, cluster_group

def presence_ratio(spike_train, duration, num_bin_edges=101):
    """Calculate fraction of time the unit is present within an epoch.

    Inputs:
    -------
    spike_train : array of spike times
    duration : length of recording (seconds)
    num_bin_edges : number of bin edges for histogram
      - total bins = num_bin_edges - 1

    Outputs:
    --------
    presence_ratio : fraction of time bins in which this unit is spiking

    """

    h, b = np.histogram(spike_train, np.linspace(0, duration, num_bin_edges))

    return np.sum(h > 0) / (num_bin_edges - 1)

def isi_violations(spike_train, duration, isi_threshold, min_isi=0):
    """Calculate Inter-Spike Interval (ISI) violations for a spike train.

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Originally written in Matlab by Nick Steinmetz (https://github.com/cortex-lab/sortingQuality)
    Converted to Python by Daniel Denman

    Inputs:
    -------
    spike_train : array of monotonically increasing spike times (in seconds) [t1, t2, t3, ...]
    duration : length of recording (seconds)
    isi_threshold : threshold for classifying adjacent spikes as an ISI violation
      - this is the biophysical refractory period
    min_isi : minimum possible inter-spike interval (default = 0)
      - this is the artificial refractory period enforced by the data acquisition system
        or post-processing algorithms

    Outputs:
    --------
    fpRate : rate of contaminating spikes as a fraction of overall rate
      - higher values indicate more contamination
    num_violations : total number of violations detected

    """
    isis_initial = np.diff(spike_train)

    if min_isi > 0:
        duplicate_spikes = np.where(isis_initial <= min_isi)[0]
        spike_train = np.delete(spike_train, duplicate_spikes + 1)

    isis = np.diff(spike_train)
    num_spikes = len(spike_train)
    num_violations = sum(isis < isi_threshold)
    violation_time = 2 * num_spikes * (isi_threshold - min_isi)
    total_rate = len(spike_train)/duration
    violation_rate = num_violations / violation_time
    fpRate = violation_rate / total_rate
    frac_violations = num_violations / num_spikes

    return fpRate, num_violations, frac_violations


def get_stim_times(data_path, behavior_timestamp):

    these_trigs = get_experiment_trigger_dict(data_path, behavior_timestamp)
    start_times = these_trigs['stim_on']
    end_times = these_trigs['stim_off']
    return start_times, end_times

def get_trial_times(data_path, behavior_timestamp):

    these_trigs = get_experiment_trigger_dict(data_path, behavior_timestamp)
    start_times = these_trigs['tr_on']
    end_times = these_trigs['tr_off']
    return start_times, end_times

def get_expt_times(data_path, behavior_timestamp): 
    """
    get start and stop time of experiment 
    """
    these_trigs = get_experiment_trigger_dict(data_path, behavior_timestamp)
    t_start = these_trigs['t_start']
    t_end = these_trigs['t_end']
    return t_start, t_end

def get_experiment_trigger_dict(data_path, behavior_timestamp): 
    
    if 'imec0' in str(data_path): 
        data_path = data_path.parent

    # # load trigger file for times 
    trigs = np.load(data_path / 'triggers.npz', allow_pickle=True)
    triggers = trigs['triggers'].item()

    trig_keys = list(triggers.keys())
    behavior_files = [x['behav_file_name'] for x in triggers.values()]

    trigidx = [i for i, s in enumerate(behavior_files) if behavior_timestamp in s][0]
    expt_triggers = triggers[trig_keys[trigidx]]
    return expt_triggers

def return_images_displayed(data_path, behavior_timestamp): 
    """
    Returns a list of images displayed, i.e. those in the image folder (not sequence of images displayed)

    Inputs: 
    data_path (Path): path to the data directory
    behavior_timestamp (str): timestamp of behavior file

    Returns: 
    images_displayed(list): list of image filenames displayed in the experiment - EXCLUDES WAKE UP IMAGES

    """

    behav = np.load(return_behav_file(data_path, behavior_timestamp), allow_pickle=True)
    calib_settings = behav['calib_settings'].item()
    images_displayed= calib_settings['img_list']
    return images_displayed

def return_image_sequence(data_path, behavior_timestamp): 
    '''
    For image mode trials, return the image sequence as a flattened array
    Returned array should equal number of stimulus triggers recorded in SpikeGLX
    Accounts for the following: 
    - Wake up images
    - Failed trial init
    - Empty image slots (e.g. RSVP trials)
    - Image sequence shape (1D sequence or 2D for RSVPs)

    Returns: 
    image_sequence (list): list of image filenames displayed in the experiment, including wakeup images
    wakeup_image_filt (numpy.ndarray): 
    rsvp_idx (numpy.ndarray): 
    '''

    behav = np.load(return_behav_file(data_path, behavior_timestamp), allow_pickle=True)
    calib = behav['calib'].item()
    image_sequence = calib['image_displayed']

    if type (image_sequence[0]) == str: 
        image_sequence = np.atleast_2d(np.array(image_sequence))
    
    # make an rsvp_idx for determining RSVP order 
    rsvps = image_sequence.shape[0]
    rsvp_idx = np.ones_like(image_sequence).astype(int)
    for ii in range(rsvps):
        rsvp_idx[ii, :] = rsvp_idx[ii, :]*(ii+1)
    
    wakeup_image_sequence = calib['wake_up_image_displayed']
    wu_flag = calib['n_wake_up_trs']>0
    if wu_flag: 
        # Create an index for non-empty elements in wakeup_image_sequence
        if isinstance(wakeup_image_sequence[0], (str, np.str_)):
            non_empty_indices = [idx for idx, val in enumerate(wakeup_image_sequence) if val.strip() and val.strip() != '[]']
        else:
            non_empty_indices = [idx for idx, val in enumerate(wakeup_image_sequence) if val]

        wakeup_image_filt = np.zeros_like(image_sequence, dtype=bool)

        # Insert non-empty filenames from wakeup_image_sequence into the first row of image_sequence
        for idx in non_empty_indices:
                #image_sequence[0, idx] = np.array(wakeup_image_sequence[idx].item())
                if isinstance(image_sequence[0,idx], (str, np.str_)): # lazy, but if image sequence is strings, keep it the same
                    if isinstance(wakeup_image_sequence[idx], (str | np.str_)): 
                        image_sequence[0, idx] = wakeup_image_sequence[idx]
                    else: 
                        image_sequence[0, idx] = wakeup_image_sequence[idx].item()
                else: 
                    image_sequence[0, idx] = wakeup_image_sequence[idx]
                wakeup_image_filt[0,idx] = True
                rsvp_idx[1:,idx] = 0 # set any row element after the first to 0, for removal later
    else: 
        wakeup_image_filt = None

    image_sequence_before_RS = image_sequence.copy()

    tr_init_failed = calib['trial_init_timed_out'].astype(bool)  # Ensure boolean type
    image_sequence = image_sequence[:, ~tr_init_failed]  # Filter columns
    rsvp_idx = rsvp_idx[:, ~tr_init_failed]
    
    if wu_flag: 
        wakeup_image_filt = wakeup_image_filt[:, ~tr_init_failed]

    # process image sequence
    if rsvps > 1: 
        image_sequence = image_sequence.transpose().reshape(-1)            
        rsvp_idx = rsvp_idx.transpose().reshape(-1)
        if wu_flag: 
            wakeup_image_filt = wakeup_image_filt.transpose().reshape(-1)
    else: 
        image_sequence = np.squeeze(image_sequence) # squeeze it back after the atleast_2d
        rsvp_idx = np.squeeze(rsvp_idx)
        if wu_flag:
            wakeup_image_filt = np.squeeze(wakeup_image_filt)
    
    #image_sequence_empty_idx = np.array([True if ((isinstance(img, str) and img.strip() == '[]') or img.size == 0) else False for img in image_sequence])
    image_sequence_empty_idx = np.array([True if ((isinstance(img, str) and img.strip() == '[]') or (not(isinstance(img, str)) and img.size == 0)) else False for img in image_sequence])
    #image_sequence = [str(img) for img in image_sequence if img.size > 0]
    image_sequence = image_sequence[image_sequence_empty_idx == False]
    wakeup_image_filt = wakeup_image_filt[image_sequence_empty_idx == False]
    rsvp_idx = rsvp_idx[image_sequence_empty_idx == False]    

    image_sequence = image_sequence.tolist()

    if isinstance(image_sequence[0], (str, np.str_)): 
        image_sequence = [Path(img).name for img in image_sequence if img]  # Ensure non-empty strings
    else: 
        image_sequence = [Path(img[0]).name for img in image_sequence if img] 
    return image_sequence, wakeup_image_filt, rsvp_idx

def filter_stim_sequence(data_path, behavior_timestamp, image_filters, use_regex=False, rm_wakeup=True): 
    """
    image_filters (list or str)
    data_path ()
    behavior_timestamp (str): timestamp of behavior file

    """
    image_sequence, wakeup_image_filt, _ = return_image_sequence(data_path, behavior_timestamp)
    stim_filters = list()
    if isinstance(image_filters, str):
        image_filters = [image_filters]
        
    for this_filt in image_filters:
        print(f'filtering for {this_filt}, use regex: {use_regex}')
        stim_filters.append(create_binary_vector(image_sequence, this_filt, use_regex=use_regex))  

    stim_filters = np.array(stim_filters)

    if rm_wakeup and wakeup_image_filt is not None:
        stim_filters[:, wakeup_image_filt] = 0

    return stim_filters


def filter_valid_eye(data_path, behavior_timestamp, valid_thresh = 50): 

    behav = np.load(return_behav_file(data_path, behavior_timestamp), allow_pickle=True)
    eye = behav['eyetrack'].item()
    perc_valid = eye['valid_filter']['perc_valid'] 
    valid_filter = perc_valid >= valid_thresh
    return valid_filter

def get_behavior_npzs(data_path): 

    if 'imec0' in str(data_path): 
        data_path = data_path.parent

    behavior_path = data_path / 'behavior_py'
    behav_flist = list(behavior_path.rglob('calib*.npz'))
    return behav_flist


def return_behav_file(data_path, behavior_timestamp): 
    behav_flist = get_behavior_npzs(data_path)
    behavior_file = [s for i, s in enumerate(behav_flist) if behavior_timestamp in str(s)]
    return behavior_file[0]


def return_behav_feature(data_path, behavior_timestamp, which_dict, which_feature=None): 

    behav = np.load(return_behav_file(data_path, behavior_timestamp), allow_pickle=True)
    this_dict = behav[which_dict].item()

    key_list = list(this_dict.keys())

    if which_feature is None: 
        print(f'No feature supplied, printing keys from dict "{which_dict}"')
        pp.pprint(key_list)
        return this_dict
    elif which_feature not in key_list:
        suggestion = fuzzy_suggest(key_list, which_feature) 
        if suggestion is not None: 
            print(f'"{which_feature}" not found, did you mean "{suggestion}"?')
        else: 
            print(f'Feature "{which_feature}" not found, printing keys from dict: "{which_dict}"')
            pp.pprint(key_list)        
        return key_list
    else: 
        this_feature = this_dict[which_feature]
        return this_feature


def fuzzy_suggest(strlist, token, thresh = 0.8): 
    ratio_score = np.zeros_like(strlist, dtype='f')
    for ii, this_str in enumerate(strlist): 
        ratio_score[ii] = fuzz.ratio(this_str, token)
    if any(ratio_score > thresh): 
        outidx = ratio_score.argmax()
        return strlist[outidx]
    else: 
        return None


def sig_resp_calc(spike_array, timewin1, timewin2, test='ttest'):
    """
    Compare firing rates for spike events in timewin1 and timewin2 and return a stats structure.

    Parameters:
        spike_array (list): list of spike trains organized by trial, output of return_spike_array
        timewin1 (list): two element list describing time window 1, e.g. [-0.2, 0]
        timewin2 (list): as above, for time window 2, e.g. [0.1, 0.3]
        test (str): either 'ttest', or 'wilcoxon', both paired tests

    Returns:
        res: scipy stats result object with 'pvalue', 'statistic' and others 
    """    
    timewin1_FR = np.zeros(len(spike_array))
    timewin2_FR = np.zeros(len(spike_array))

    for ii, spikes_trial in enumerate(spike_array):  
        timewin1_FR[ii] = (np.sum((spikes_trial > timewin1[0]) & (spikes_trial<=timewin1[1]))/np.diff(timewin1))[0]
        timewin2_FR[ii] = (np.sum((spikes_trial > timewin2[0]) & (spikes_trial<=timewin2[1]))/np.diff(timewin2))[0]

    if test == 'ttest': 
        res = stats.ttest_rel(timewin1_FR, timewin2_FR)
    elif test == 'wilcoxon': 
        res = stats.wilcoxon(timewin1_FR, timewin2_FR)

    return res



def ret_waveforms_best_ch(cluster_id: int, spikesort_path: str | Path, n_spikes=500, smooth_wfs=True, spike_train=None): 
    """
    Return spike waveforms for cluster_id

    Parameters:
        cluster_id (int): 
        spikesort_dir (str or Path): 
        nr_spikes (int): [500] returns wfs for n_spikes or all spikes if total spike count < n_spikes
        smooth_wfs (bool): [True] apply a Savitzky-Golay filter to the spike waveforms 

    Returns:
        wfs
        best_ch (int)
    """

    spikesort_path = Path(spikesort_path)
    #t5 = getsecs()
    if spike_train is None:
        cluster_spks = get_cluster_spikes(cluster_id, spikesort_path, n_spikes=n_spikes)
    #t6 = getsecs()
    #print(f'get_cluster_spikes took {t6-t5} secs')
    #print(f'len cluster spks: {len(cluster_spks)}')
    templates_ind_file = spikesort_path / 'templates_ind.npy'
    templates_ch_ind = np.load(templates_ind_file, allow_pickle=True)

    # ops = load_ops(spikesort_path / 'ops.npy')
    # recording_dat_file = spikesort_path / 'recording.dat'
    # if recording_dat_file.exists(): 
    #     bfile = bfile_from_ops(ops=ops, filename=str(spikesort_path / 'recording.dat'))
    #     wfs = get_spike_waveforms(cluster_spks, str(spikesort_path), bfile=bfile)
    # else: 
    #     wfs = get_spike_waveforms(cluster_spks, str(spikesort_path))
    # best_ch_ind = get_best_waveform_ch(wfs)
    # wfs = wfs[best_ch_ind,:,:]
    # best_ch = templates_ch_ind[cluster_id, best_ch_ind]
    
    try: 
        ops = load_ops(spikesort_path / 'ops.npy')
        if spike_train is not(None):
            cluster_spks = np.array(spike_train * ops['fs']).astype(int)
        recording_dat_file = spikesort_path / 'recording.dat'
        if recording_dat_file.exists(): 
            #t1 = getsecs()
            bfile = bfile_from_ops(ops=ops, filename=str(spikesort_path / 'recording.dat'))
            #t2 = getsecs()
            #print(f'load bfile took {t2 -t1} secs')
            t3 = getsecs()
            wfs = get_spike_waveforms(cluster_spks, spikesort_path, bfile=bfile)
            t4 = getsecs()
            print(f'get_spike_waveforms took {t4-t3 :.4f} secs')
        else: 
            wfs = get_spike_waveforms(cluster_spks, spikesort_path)
        best_ch_ind = get_best_waveform_ch(wfs)
        wfs = wfs[best_ch_ind,:,:]
        best_ch = templates_ch_ind[cluster_id, best_ch_ind]
        if smooth_wfs:
            wfs = smooth_signal(wfs)
        return wfs, best_ch
    except Exception as e: 
        print('ret_wavforms_best_ch: error returning waveforms:')
        print(e)
        print('will try templates instead')
        print('this will return only the "best channel" template (highest amplitude)')
        templates_file = spikesort_path / 'templates.npy'
        templates = np.load(templates_file, allow_pickle=True)

        best_ch_ind = get_best_template_ch(templates[cluster_id,:,:])
        best_ch = templates_ch_ind[cluster_id,best_ch_ind]
        print(f'returning best template on ch {best_ch}')
        best_template = templates[cluster_id,:,best_ch_ind]
        return best_template, best_ch


def get_best_template_ch(templates): 
    """
    Takes a matrix of templates from a single cluster 
    n_timepoints x n_chans (as loaded by )
    """
    assert templates.ndim == 2
    peak_values = np.max(np.abs(templates), axis = 0)
    best_ch_ind = np.argmax(peak_values)
    return best_ch_ind    

def get_best_waveform_ch(waveforms):
    """
    Return best waveform channel, i.e. channel on which median waveforms is biggest
    This may be preferable to the kilosort API function get_best_channel - does not seem to return the correct best ch 

    Parameters:
        waveforms (np array)

    Returns:
        best_ch (int)
    """
    mean_wfs_calc = np.median(waveforms, axis=2)
    peak_values = np.max(np.abs(mean_wfs_calc),axis=1)
    best_ch_ind = np.argmax(peak_values)
    return best_ch_ind

def plot_waveforms(wfs, col='green', meancol='blue', n_plot=None, alpha=0.1, plot_mean=True): 
    
    if n_plot is None:
        plt.plot(wfs, col, alpha=alpha)
    if plot_mean: 
        plt.plot(np.median(wfs, axis= 1), color=meancol, linestyle='--', label='median WF')
    plt.autoscale(enable=True, tight=True)
    box_off(plt.gca())


def plot_unit_over_session(spiketrain, spikesort_path, binsz=60, color='green', make_plot=True,):
    ops = load_ops(Path(spikesort_path) / 'ops.npy')
    total_dur = get_recording_duration(ops, spikesort_path)
    #print(f'recording duration is {total_dur: .2f} sec')
    nbins = int(np.ceil(np.ceil(total_dur/binsz)))
    bin_edges = np.linspace(0, nbins*binsz, nbins+1)
    hist,_ = np.histogram(spiketrain, bin_edges)
    bin_cents = np.diff(bin_edges)/2+bin_edges[:-1]
    #print(bin_edges)
    if make_plot:  # otherwise just return the histogram
        plt.bar(bin_cents, hist, width=binsz, color=color); 
        plt.ticklabel_format(style='sci', scilimits=(0,3))
        plt.xlabel('Time (s)')
        plt.ylabel('Count')
        plt.autoscale(enable=True, tight=True,axis='x')
        xL=plt.xlim()
        yL=plt.ylim()
        xloc_text = np.diff(xL)*0.8+xL[0]
        yloc_text = np.diff(yL)*0.9+yL[0]
        plt.text(x=xloc_text,y=yloc_text,s=f'{binsz}s bins', fontsize=8)
        plt.title('Unit activity over recording session', \
                  fontsize=9)

    return hist, bin_cents

def get_recording_duration(ops, spikesort_path=None): 
    """
    Return recording duration in seconds

    Paramaters: 
    ops - kilosort ops, returned from load_ops
    Returns: 
    recording_dur (float) in seconds
    """
    try: 
        data_file = ops['data_file_path']
    except Exception as e: 
        #print(Path(spikesort_path) / 'recording.dat')
        assert spikesort_path is not None
        data_file = Path(spikesort_path) / 'recording.dat'    
    dtype = ops['data_dtype']
    n_chans = ops['n_chan_bin']
    n_samps = get_total_samples(data_file, n_chans, dtype=dtype)
    recording_dur = n_samps/ops['fs']
    return recording_dur 

def filter_clusts_fr(clustlist, spikesort_path, start_end_t = None, min_FR=0.1): 
    """
    Parameters: 
    clustlist (list or np array): 
    data_path (pathLib obj):  for data e.g. r'D:\SpikeGLX\Cashew_20250128_g0\Cashew_20250128_g0_imec0'
    spikesort_dir (str): directory name
    start_end_t (tuple or list): default of None filters by all spikes in a recording session
    min_FR (float): spks/sec value default [0.1 spks/sec]

    Returns: 

    """
    clustlist = np.atleast_1d(clustlist)

    if start_end_t is None: 
        ops = load_ops(spikesort_path / 'ops.npy')
        total_dur = get_recording_duration(ops, spikesort_path)
        expt_start = 0
        expt_end = total_dur
    else: 
        expt_start = start_end_t[0]
        expt_end = start_end_t[1]
    st, sc, cg = get_spikes(spikesort_path, which_clusts=clustlist)
    fr_curr = np.zeros(len(clustlist))
    for ii, clust in enumerate(clustlist):
        st_curr = st[sc==clust]
        fr_curr[ii] = int(np.sum((st_curr>=expt_start) & (st_curr<expt_end)))/(expt_end-expt_start)
    
    clustlist_filt = clustlist[fr_curr>=min_FR]
    frs_filt = fr_curr[fr_curr>=min_FR]

    return clustlist_filt, frs_filt

def waveform_peak_trough_time(waveform, sampling_rate_khz=30): 
    # Find peak (largest absolute amplitude)
    peak_idx = np.argmax(np.abs(waveform))
    
    # Define search window after peak to find trough
    if waveform[peak_idx] > 0:
        trough_idx = peak_idx + np.argmin(waveform[peak_idx:])
    else:
        trough_idx = peak_idx + np.argmax(waveform[peak_idx:])
    
    # Calculate time difference (in samples) and convert to ms
    peak_trough_samples = trough_idx - peak_idx
    peak_trough_time_ms = peak_trough_samples / sampling_rate_khz
    return peak_trough_time_ms, peak_idx, trough_idx