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
        if len(trial_filt) != len(trig_times): 
            print('WARNING: trial_filt length does not match trig_times length')
            print(f'trial_filt length: {len(trial_filt)}; trig_times length: {len(trig_times)}')
            print('truncating trial_filt to match trig_times length')
            trial_filt = trial_filt[:len(trig_times)]
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
    """
    Calculate PSTH from spike array

    Parameters:
    spike_array (list of numpy ndarrays)
    t1 (float)
    t2 (float)
    binwidth (float)

    Returns:
    psth_mu (numpy array)
    psth_std (numpy array)
    psth_sem (numpy array)
    time (numpy array)
    
    """
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
            struct_dict[key] = val
            #exec(f"struct_dict['{key}'] = {key}")
        
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
                if data_out[nonempty_idx].size == 0:
                    data_out = []
                elif type(data_out[nonempty_idx][0]) is np.str_: 
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

    #trigidx = [i for i, s in enumerate(behavior_files) if behavior_timestamp in s][0]
    trigidx = [i for i, s in enumerate(behavior_files) if s is not None and behavior_timestamp.strip() in s][0]
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
    behavior_file = [s for i, s in enumerate(behav_flist) if behavior_timestamp.strip() in str(s)]
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
    plt.grid(False)
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
        fr_curr[ii] = np.sum((st_curr>=expt_start) & (st_curr<expt_end))/(expt_end-expt_start)
    
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


# ================================================================================
# Active Period Detection Functions
# ================================================================================

def detect_active_periods(spike_train, total_duration, bin_size=60, step_size=30,
                         threshold_percentile=20, min_consecutive_bins=2,
                         max_gap_to_merge=120, use_high_percentile=True,
                         high_percentile=75):
    """
    Detect active and inactive periods for a single unit based on firing rate.

    Uses a sliding window approach with a forgiving threshold to identify
    periods when the unit stops firing or has very low activity. Includes
    gap merging to handle occasional blips during dropout periods, and uses
    a high percentile for threshold calculation to handle units with brief
    high-activity periods.

    Parameters
    ----------
    spike_train : np.ndarray
        Array of spike times in seconds
    total_duration : float
        Total recording duration in seconds
    bin_size : float, optional
        Size of sliding window in seconds (default: 60)
    step_size : float, optional
        Step size for sliding window in seconds (default: 30)
    threshold_percentile : float, optional
        Percentage of baseline firing rate below which a bin is considered
        inactive (default: 20, meaning 20% of baseline)
    min_consecutive_bins : int, optional
        Minimum number of consecutive low-FR bins to mark as inactive
        (default: 2, for sustained dropouts)
    max_gap_to_merge : float, optional
        Maximum gap (in seconds) between inactive periods to merge them.
        Helps eliminate striping from occasional blips during dropouts.
        (default: 120 seconds)
    use_high_percentile : bool, optional
        If True, use high_percentile of firing rates as baseline instead of
        median. This helps detect dropouts in units with brief high-activity
        periods. (default: True)
    high_percentile : float, optional
        Percentile to use as baseline when use_high_percentile=True
        (default: 75, meaning 75th percentile)

    Returns
    -------
    inactive_periods : list of tuple
        List of (start_time, end_time) tuples marking inactive periods

    Examples
    --------
    >>> inactive = detect_active_periods(spike_times, 3600, bin_size=60, step_size=30)
    >>> print(f"Unit was inactive during {len(inactive)} periods")

    >>> # For units with very brief high-activity periods
    >>> inactive = detect_active_periods(spike_times, 3600, use_high_percentile=True, high_percentile=90)
    """
    if len(spike_train) == 0:
        # No spikes at all - entire recording is inactive
        return [(0, total_duration)]

    # Create sliding windows
    bin_starts = np.arange(0, total_duration - bin_size + step_size, step_size)
    bin_ends = bin_starts + bin_size

    # Calculate firing rate in each bin
    firing_rates = np.zeros(len(bin_starts))
    for i, (start, end) in enumerate(zip(bin_starts, bin_ends)):
        n_spikes = np.sum((spike_train >= start) & (spike_train < end))
        firing_rates[i] = n_spikes / bin_size

    # Compute threshold using more robust baseline
    if use_high_percentile:
        # Use high percentile to avoid low overall median for brief high-FR units
        baseline_fr = np.percentile(firing_rates, high_percentile)
    else:
        baseline_fr = np.median(firing_rates)

    threshold = (threshold_percentile / 100.0) * baseline_fr

    # Identify low-FR bins
    low_fr_bins = firing_rates < threshold

    # Find consecutive stretches of low-FR bins
    inactive_periods = []
    in_inactive = False
    inactive_start = None
    consecutive_count = 0

    for i, is_low in enumerate(low_fr_bins):
        if is_low:
            if not in_inactive:
                inactive_start = bin_starts[i]
                consecutive_count = 1
                in_inactive = True
            else:
                consecutive_count += 1
        else:
            if in_inactive:
                # End of inactive period - only add if long enough
                if consecutive_count >= min_consecutive_bins:
                    inactive_end = bin_starts[i-1] + bin_size
                    inactive_periods.append((inactive_start, inactive_end))
                in_inactive = False
                consecutive_count = 0

    # Handle case where recording ends during inactive period
    if in_inactive and consecutive_count >= min_consecutive_bins:
        inactive_periods.append((inactive_start, total_duration))

    # Merge inactive periods separated by short gaps (eliminates striping)
    if len(inactive_periods) > 1 and max_gap_to_merge > 0:
        merged_periods = []
        current_start, current_end = inactive_periods[0]

        for next_start, next_end in inactive_periods[1:]:
            gap = next_start - current_end
            if gap <= max_gap_to_merge:
                # Merge: extend current period to include the gap and next period
                current_end = next_end
            else:
                # Gap too large: save current period and start new one
                merged_periods.append((current_start, current_end))
                current_start, current_end = next_start, next_end

        # Add the last period
        merged_periods.append((current_start, current_end))
        inactive_periods = merged_periods

    return inactive_periods


def load_active_periods(spikesort_path, cluster_id):
    """
    Load pre-computed inactive period data for a specific cluster.

    Parameters
    ----------
    spikesort_path : str or Path
        Path to Kilosort output directory
    cluster_id : int
        Cluster ID to retrieve

    Returns
    -------
    inactive_periods : list of tuple or None
        List of (start_time, end_time) tuples for inactive periods,
        or None if file doesn't exist or cluster not found

    Examples
    --------
    >>> inactive = load_active_periods(spikesort_path, 42)
    >>> if inactive:
    ...     print(f"Cluster 42 has {len(inactive)} inactive periods")
    """
    import pandas as pd

    spikesort_path = Path(spikesort_path)
    active_periods_file = spikesort_path / 'active_periods.pkl'

    if not active_periods_file.exists():
        return None

    try:
        df = pd.read_pickle(active_periods_file)
        matching = df[df['cluster_id'] == cluster_id]

        if len(matching) == 0:
            return None

        inactive_periods = matching.iloc[0]['inactive_periods']
        return inactive_periods if len(inactive_periods) > 0 else None

    except Exception as e:
        print(f"Warning: Could not load active periods: {e}")
        return None


def plot_inactive_bands_continuous(ax, inactive_periods, color='red', alpha=0.3):
    """
    Overlay inactive period bands on a continuous-time plot.

    For plots where time is on the X-axis (e.g., activity histogram over session).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    inactive_periods : list of tuple
        List of (start_time, end_time) tuples marking inactive periods
    color : str, optional
        Color for the bands (default: 'red')
    alpha : float, optional
        Transparency level (default: 0.3)

    Returns
    -------
    patches : list
        List of matplotlib patch objects added to the axes

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> plot_unit_over_session(spike_train, spikesort_path)
    >>> inactive = [(100, 200), (500, 600)]
    >>> plot_inactive_bands_continuous(ax, inactive)
    """
    patches = []
    for start, end in inactive_periods:
        patch = ax.axvspan(start, end, facecolor=color, alpha=alpha, zorder=2)
        patches.append(patch)
    return patches


def get_inactive_trials(inactive_periods, trial_times):
    """
    Determine which trials occurred during inactive periods.

    Parameters
    ----------
    inactive_periods : list of tuple
        List of (start_time, end_time) tuples marking inactive periods
    trial_times : np.ndarray
        Array of trial trigger times in seconds

    Returns
    -------
    inactive_trials : np.ndarray
        Boolean array (length = n_trials) where True indicates the trial
        occurred during an inactive period

    Examples
    --------
    >>> inactive_periods = [(100, 200), (500, 600)]
    >>> trial_times = np.array([50, 150, 250, 550, 700])
    >>> inactive = get_inactive_trials(inactive_periods, trial_times)
    >>> # Returns: [False, True, False, True, False]
    """
    if inactive_periods is None or len(inactive_periods) == 0:
        return np.zeros(len(trial_times), dtype=bool)

    inactive_trials = np.zeros(len(trial_times), dtype=bool)

    for start, end in inactive_periods:
        # Mark trials whose trigger time falls within this inactive period
        in_period = (trial_times >= start) & (trial_times < end)
        inactive_trials |= in_period

    return inactive_trials


def plot_inactive_bands_raster(ax, inactive_trial_indices, color='red', alpha=0.3):
    """
    Overlay inactive trial bands on a trial-based raster plot.

    For raster plots where trials are on the Y-axis. Draws horizontal bands
    across specific trial rows.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    inactive_trial_indices : np.ndarray or list
        Indices of trials (0-based) that should be marked as inactive
    color : str, optional
        Color for the bands (default: 'red')
    alpha : float, optional
        Transparency level (default: 0.3)

    Returns
    -------
    patches : list
        List of matplotlib patch objects added to the axes

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> plot_raster(spike_array, -0.2, 0.5, ax=ax)
    >>> inactive_indices = [5, 6, 12, 13, 14]  # trials 5, 6, 12-14 are inactive
    >>> plot_inactive_bands_raster(ax, inactive_indices)
    """
    patches = []

    # Convert to array if needed
    if isinstance(inactive_trial_indices, (list, np.ndarray)):
        inactive_trial_indices = np.array(inactive_trial_indices)

    # If it's a boolean array, convert to indices
    if inactive_trial_indices.dtype == bool:
        inactive_trial_indices = np.where(inactive_trial_indices)[0]

    for trial_idx in inactive_trial_indices:
        # Draw horizontal band for this trial
        # Y-coordinates in raster are typically 0-indexed trial numbers
        patch = ax.axhspan(trial_idx, trial_idx + 1,
                          facecolor=color, alpha=alpha, zorder=1)
        patches.append(patch)

    return patches


def return_PSTH_with_active_filter(spike_array, trial_active_filter, t1, t2, binwidth):
    """
    Calculate PSTH from spike array, excluding inactive trials.

    Modified version of return_PSTH that only includes trials marked as active.
    This prevents dropout periods from artificially lowering the mean firing rate.

    Parameters
    ----------
    spike_array : list of np.ndarray
        Output of return_spike_array - list of spike time arrays per trial
    trial_active_filter : np.ndarray
        Boolean array (length = n_trials) where True indicates an active trial
    t1 : float
        Start of PSTH window (seconds relative to trigger)
    t2 : float
        End of PSTH window
    binwidth : float
        Bin width in seconds (e.g., 0.01 for 10 ms bins)

    Returns
    -------
    psth_mu : np.ndarray
        Mean firing rate per bin (Hz), computed only from active trials
    psth_std : np.ndarray
        Standard deviation across active trials
    psth_sem : np.ndarray
        Standard error of the mean (uses n_active_trials)
    time : np.ndarray
        Bin center times

    Examples
    --------
    >>> # Get spike array and determine active trials
    >>> spike_array = return_spike_array(st, triggers, [-0.2, 0.5])
    >>> active = ~get_inactive_trials(inactive_periods, triggers)
    >>> mu, std, sem, t = return_PSTH_with_active_filter(spike_array, active,
    ...                                                   -0.2, 0.5, 0.01)
    """
    # Filter spike array to only include active trials
    active_spikes = [spike_array[i] for i in range(len(spike_array))
                     if trial_active_filter[i]]

    n_active_trs = len(active_spikes)

    if n_active_trs == 0:
        # No active trials - return zeros
        psth_bins = np.arange(t1, t2 + binwidth, binwidth)
        n_bins = len(psth_bins) - 1
        return (np.zeros(n_bins), np.zeros(n_bins),
                np.zeros(n_bins), np.diff(psth_bins)/2 + psth_bins[:-1])

    # Compute PSTH using only active trials
    psth_bins = np.arange(t1, t2 + binwidth, binwidth)
    spike_hist_all = np.array([np.histogram(spikes_trial, psth_bins)[0]
                               for spikes_trial in active_spikes])
    spike_hist_all = spike_hist_all / binwidth

    psth_mu = np.mean(spike_hist_all, axis=0)
    psth_std = np.std(spike_hist_all, axis=0)
    psth_sem = psth_std / np.sqrt(n_active_trs)
    time = np.diff(psth_bins) / 2 + psth_bins[:-1]

    return psth_mu, psth_std, psth_sem, time