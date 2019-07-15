# -*- coding: utf-8 -*-
"""
Binary file parser for 500MHz oscilliscope data
of PMT/MCP/Silicon PM. The purpose is also to
sort and find the after pulses for further analysis
"""
import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
import matplotlib as mpl
mpl.rcParams['font.size'] = 14                         # Set the general plotting font size                                 # Make the plots nicer to look at
import pandas as pd
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
from probfit import Chi2Regression                     # Helper tool for fitting
from scipy import stats
from scipy.interpolate import interp1d
np.random.seed(42)
global threshold, max_signal_amplitude
threshold = -0.032 # V maybe change 
max_signal_amplitude = -0.2 # V
global ALICE_deadtime, start_time
ALICE_deadtime, start_time = 25, 770

def GetBinaryLength(filename, EventsTotal):
    """ Binary files can be tricky to read, so
    this function returns the approximately size of
    the data of one event. This can be used as a tool
    for reading the file or to cross check whether
    the binary file was read correctly """
    with open(filename, 'rb') as fid:
        BigData = np.fromfile(fid, np.single)

    global sigma, grass
    sigma = np.std(BigData[:100])
    grass = np.mean(BigData[:100])

    return int(len(BigData)/EventsTotal)

def quad_GaussianFit(Nbins,xmin,xmax,data):
    """Given inputs of data and corresponding histogram
    returns Chi2 regression Gaussian fit from Minuit"""
    fig, ax = plt.subplots()
    hist, bins, _ = ax.hist(data, bins=Nbins, range=(xmin,xmax))
    plt.close(fig)
    
    
    # Defining variables 
    dis_bins = bins[1] - bins[0]
    x_hist = np.arange(xmin+dis_bins/2,(xmax),dis_bins)
    x_var =  np.linspace(xmin-dis_bins/2,(xmax),1000)
    
    # Only inputs from bins >0
    histcor = hist[hist>0]
    x_hist = x_hist[hist>0]
    error = np.array([1/i for i in histcor])
    N_scale =  max(hist)/max(gauss_pdf(x_var, np.mean(data), np.std(data)))
    
    # Fit
    chi2_object = Chi2Regression(quad_gauss, x_hist, histcor, weights=error) 
    minuit = Minuit(chi2_object, pedantic=False, 
                    N1=N_scale, N2=N_scale/2, N3=N_scale/3, N4=N_scale/4,
                    mu1=15, mu2=20, mu3=35, mu4=40,
                    sigma1=10, sigma2=5, sigma3=5, sigma4=5, 
                    limit_N1=(1,2000), limit_N2=(1,2000), 
                    limit_N3=(1,2000), limit_N4=(1,2000),
                    limit_mu1=(min(data),100), limit_mu2=(min(data),100), 
                    limit_mu3=(min(data),100), limit_mu4=(min(data),100), 
                    limit_sigma1=(1,8), limit_sigma2=(1,8), 
                    limit_sigma3=(1,8), limit_sigma4=(1,10), 
                    print_level=0)
    
    minuit.migrad();   
    NDOF = len(histcor) - 12
    Chi2_fit = minuit.fval
    Prob_fit = stats.chi2.sf(Chi2_fit, NDOF)
    
    #Parameters of fit
    (N1_fit, N2_fit, N3_fit, N4_fit,
     mean1_fit, mean2_fit, mean3_fit, mean4_fit,
     sigma1_fit, sigma2_fit, sigma3_fit, sigma4_fit) = minuit.args
    (N1_fite, N2_fite, N3_fite, N4_fite,
     mean1_fite, mean2_fite, mean3_fite, mean4_fite,
     sigma1_fite, sigma2_fite, sigma3_fite, sigma4_fite) = minuit.errors.values()
    

    
    # Text string to use as a label or info box later
    text_string = (r"Over all gaussian $\chi^2$ in numbers:" + "\n"
                    + r"$\chi^2$ = " + str(round(Chi2_fit,2)) + "\n"
                    + "NDOF = " + str(NDOF) + "\n"
                    + "Prob = " + str(round(Prob_fit*100,3)) + "%")
    
    return minuit, x_var, quad_gauss(x_var, *minuit.args), text_string

def triple_GaussianFit(Nbins,xmin,xmax,data):
    """Given inputs of data and corresponding histogram
    returns Chi2 regression Gaussian fit from Minuit"""
    fig, ax = plt.subplots()
    hist, bins, _ = ax.hist(data, bins=Nbins, range=(xmin,xmax))
    plt.close(fig)
    
    
    # Defining variables 
    dis_bins = bins[1] - bins[0]
    x_hist = np.arange(xmin+dis_bins/2,(xmax),dis_bins)
    x_var =  np.linspace(xmin-dis_bins/2,(xmax),1000)
    
    # Only inputs from bins >0
    histcor = hist[hist>0]
    x_hist = x_hist[hist>0]
    error = np.array([1/i for i in histcor])
    N_scale =  max(hist)/max(gauss_pdf(x_var, np.mean(data), np.std(data)))
    
    # Fit
    chi2_object = Chi2Regression(triple_gauss, x_hist, histcor, weights=error) 
    minuit = Minuit(chi2_object, pedantic=False, 
                    N1=N_scale, N2=N_scale/2, N3=N_scale/3,
                    mu1=15, mu2=20, mu3=35,
                    sigma1=10, sigma2=5, sigma3=5,
                    print_level=0)
    
    minuit.migrad();   
    NDOF = len(histcor) - 9
    Chi2_fit = minuit.fval
    Prob_fit = stats.chi2.sf(Chi2_fit, NDOF)
    
    #Parameters of fit
    (N1_fit, N2_fit, N3_fit, 
     mean1_fit, mean2_fit, mean3_fit, 
     sigma1_fit, sigma2_fit, sigma3_fit) = minuit.args
    (N1_fite, N2_fite, N3_fite, 
     mean1_fite, mean2_fite, mean3_fite, 
     sigma1_fite, sigma2_fite, sigma3_fite) = minuit.errors.values()
    
    # Significant decimals
    decmu1, decstd1 = decimals(mean1_fite), decimals(sigma1_fite)
    
    # Text string to use as a label or info box later
    text_string = (r"Gaussian $\chi^2$ in numbers:" + "\n"
                    + r"$\mu$ = " + str(round(mean1_fit,decmu1)) + 
                    r"$\pm$" + str(round(mean1_fite,decmu1)) + "\n"
                    + r"$\sigma$ = " + str(round(sigma1_fit,decstd1)) + 
                    r"$\pm$" + str(round(sigma1_fite,decstd1)) + "\n"
                    + r"$\chi^2$ = " + str(round(Chi2_fit,2)) + "\n"
                    + "NDOF = " + str(NDOF) + "\n"
                    + "Prob = " + str(round(Prob_fit*100,3)) + "%")
    
    return minuit, x_var, triple_gauss(x_var, *minuit.args), text_string

def double_GaussianFit(Nbins,xmin,xmax,data):
    """Given inputs of data and corresponding histogram
    returns Chi2 regression Gaussian fit from Minuit"""
    fig, ax = plt.subplots()
    hist, bins, _ = ax.hist(data, bins=Nbins, range=(xmin,xmax))
    plt.close(fig)
    
    
    # Defining variables 
    dis_bins = bins[1] - bins[0]
    x_hist = np.arange(xmin+dis_bins/2,(xmax),dis_bins)
    x_var =  np.linspace(xmin-dis_bins/2,(xmax),1000)
    
    # Only inputs from bins >0
    histcor = hist[hist>0]
    x_hist = x_hist[hist>0]
    error = np.array([1/i for i in histcor])
    N_scale =  max(hist)/max(gauss_pdf(x_var, np.mean(data), np.std(data)))
    
    # Fit
    chi2_object = Chi2Regression(double_gauss, x_hist, histcor, weights=error) 
    minuit = Minuit(chi2_object, pedantic=False, 
                    N1=N_scale, N2=N_scale,
                    mu1=np.mean(data), mu2=np.mean(data),
                    sigma1=np.std(data), sigma2=np.std(data),
                    print_level=0)
    
    minuit.migrad();   
    NDOF = len(histcor) - 6
    Chi2_fit = minuit.fval
    Prob_fit = stats.chi2.sf(Chi2_fit, NDOF)
    
    #Parameters of fit
    N1_fit, N2_fit, mean1_fit, mean2_fit, sigma1_fit, sigma2_fit = minuit.args
    N1_fite, N2_fite, mean1_fite, mean2_fite, sigma1_fite, sigma2_fite = minuit.errors.values()
    
    # Significant decimals
    decmu1, decstd1 = decimals(mean1_fite), decimals(sigma1_fite)
    
    # Text string to use as a label or info box later
    text_string = (r"Gaussian $\chi^2$ in numbers:" + "\n"
                    + r"$\mu$ = " + str(round(mean1_fit,decmu1)) + 
                    r"$\pm$" + str(round(mean1_fite,decmu1)) + "\n"
                    + r"$\sigma$ = " + str(round(sigma1_fit,decstd1)) + 
                    r"$\pm$" + str(round(sigma1_fite,decstd1)) + "\n"
                    + r"$\chi^2$ = " + str(round(Chi2_fit,2)) + "\n"
                    + "NDOF = " + str(NDOF) + "\n"
                    + "Prob = " + str(round(Prob_fit*100,3)) + "%")
    
    return minuit, x_var, double_gauss(x_var, *minuit.args), text_string

def GaussianFit(Nbins,xmin,xmax,data):
    """Given inputs of data and corresponding histogram
    returns Chi2 regression Gaussian fit from Minuit"""
    fig, ax = plt.subplots()
    hist, bins, _ = ax.hist(data,bins=Nbins,range=(xmin,xmax))
    plt.close(fig)
    
    
    # Defining variables 
    dis_bins = bins[1] - bins[0]
    x_hist = np.arange(xmin+dis_bins/2,(xmax),dis_bins)
    x_var =  np.linspace(xmin-dis_bins/2,(xmax),1000)
    
    # Only inputs from bins >0
    histcor = hist[hist>0]
    x_hist = x_hist[hist>0]
    error = np.array([1/i for i in histcor])
    N_scale =  max(hist)/max(gauss_pdf(x_var, np.mean(data), np.std(data)))
    
    # Fit
    chi2_object = Chi2Regression(gauss_extended, x_hist, histcor, weights=error) 
    minuit = Minuit(chi2_object, pedantic=False, N=N_scale, mu=np.mean(data),
                sigma=np.std(data), print_level=0)
    minuit.migrad();   
    #minuit.hesse();  
    NDOF = len(histcor) - 3
    Chi2_fit = minuit.fval
    Prob_fit = stats.chi2.sf(Chi2_fit, NDOF)
    
    #Parameters of fit
    N_fit, mean_fit, sigma_fit = minuit.args
    N_fite, mean_fite, sigma_fite = minuit.errors.values()
    
    
    # Significant decimals
    decmu, decstd = decimals(mean_fite), decimals(sigma_fite)
    
    # Text string to use as a label or info box later
    text_string = (r"Gaussian $\chi^2$ in numbers:" + "\n"
                    + r"$\mu$ = " + str(round(mean_fit,decmu)) + 
                    r"$\pm$" + str(round(mean_fite,decmu)) + "\n"
                    + r"$\sigma$ = " + str(round(sigma_fit,decstd)) + 
                    r"$\pm$" + str(round(sigma_fite,decstd)) + "\n"
                    + r"$\chi^2$ = " + str(round(Chi2_fit,2)) + "\n"
                    + "NDOF = " + str(NDOF) + "\n"
                    + "Prob = " + str(round(Prob_fit*100,3)) + "%")
    
    return minuit, x_var, gauss_extended(x_var, *minuit.args), text_string

def gauss_extended(x, N, mu, sigma):
    """Non-normalized Gaussian"""
    return N * gauss_pdf(x, mu, sigma)

def gauss_pdf(x, mu, sigma):
    """Normalized Gaussian"""
    return (1. / np.sqrt(2. * np.pi) / sigma * np.exp(-(x - mu)
                         ** 2. / 2. / sigma ** 2.))

def double_gauss(x, N1, N2, mu1, mu2, sigma1, sigma2):
    """ 2 Gaussian curves """
    return (gauss_extended(x, N1, mu1, sigma1) 
            + gauss_extended(x, N2, mu2, sigma2)) 

def triple_gauss(x, N1, N2, N3, mu1, mu2, mu3, sigma1, sigma2, sigma3):
    """ 3 Gaussian curves """
    return (gauss_extended(x, N1, mu1, sigma1) 
            + gauss_extended(x, N2, mu2, sigma2)
            + gauss_extended(x, N3, mu3, sigma3)) 
    
def quad_gauss(x, N1, N2, N3, N4, mu1, mu2, mu3, mu4, sigma1, sigma2, sigma3,
               sigma4):
    """ 4 Gaussian curves """
    return (gauss_extended(x, N1, mu1, sigma1) 
            + gauss_extended(x, N2, mu2, sigma2)
            + gauss_extended(x, N3, mu3, sigma3)
            + gauss_extended(x, N4, mu4, sigma4)) 
    
def decimals(n):
    """Returned amount of decimals
    for rounding purposes"""
    i = 0 
    if n == 0.:
        n = 1
    while n < 1:
        n = 10*n
        i += 1
    return i 

def event_integral(wave):
    """ Get integral under peak above 90% of peak, 
    and since this is a negative polarized PMT,
    a minus sign is put in front """
    return - sum(wave[wave < min(wave)*0.9])


def peak_finder(wave, top_index):
    """ From index of peak top, it finds
    and returns the isolated peak """
    N_data = len(wave)   
    
    # Left side
    for i in range(top_index):
        if wave[top_index - i] > (grass - 2*sigma):
            left_index = i
            break
    
    try: 
        left_index
    except NameError:
        left_index = 0 
    
    # Right side
    for i in range(N_data - top_index):
        if wave[top_index + i] > (grass - 2*sigma):
            right_index = i
            break
    try: 
        right_index
    except NameError:
        right_index = len(wave) - top_index
    
    peak = wave[top_index - left_index: top_index + right_index]
    
    return peak, left_index, right_index

def ImportBinary(filenames, EventLength, Events):
    """ This function will find the signal and
    afterpulse(s) of the binary data file and
    return both accompanied by the amplitude
    of afterpules, amplitude ratio between 
    afterpulse and signal, time of afterpulse and
    time delay of afterpulse """
    
    time = np.linspace(-770, 9230, EventLength)
    
    afterpulse_times = []
    amp_afterpulses, int_afterpulses = [], []
    time_arrival, integrals = [], []
    peaks, peak_class = [], []
    rate = []
    
    amp_ratio = []
    show_error = True
    global record_time
    record_time = 10000 #ns     20/3 = 1000 #ns
    max_N_afterpulses = 10
    max_show_afterpulses = 10
    shown_afterpulses = 0
    
    
    time_between_points = record_time / EventLength
    time_now = 0
    file_number = 0

    for file in filenames:
        N_signals, N_afterpulse = 0, 0
        error_ite = -1
        with open(file, 'rb') as fid:
            for i in range(0,Events[file_number]):
                data_array = np.fromfile(fid, np.single, EventLength)
                
                # If no more datapoints are read
                if len(data_array) == 0:
                    break                
                
                # Electric noise
                if max(data_array) > 0.05:
                    if show_error:
                        plt.figure()
                        length = len(data_array[10:(len(data_array)-10)])
                        times = np.array(list(range(length)))*time_between_points
                        plt.plot(times,data_array[10:(len(data_array)-10)])
                        plt.title("This 'event' was removed from analysis, since"
                                  + " it is nothing but electric noise")
                        show_error = False
                    continue
                    
                
                signal_index = np.argmin(data_array) 
                signal, left_i, right_i = peak_finder(data_array, signal_index)
                
                if len(signal) < 100:
                    continue
                
                # If highest peak isn't the one that is triggered then look
                # aside from this event
#                if time[signal_index] > 30 or time[signal_index] < -30:
#                    continue
                
                afterpulses_dict = {}
                afterpulses_right, afterpulses_left, afterpulses_i = [], [], []
                
                for l in range(max_N_afterpulses):
                    # Temporary wave without previous peaks
                    wave_temp = data_array[signal_index + right_i:]
                    
                    # Visualizing errors of wave_temp in the errors folder
                    if len(wave_temp) == 0:
                        fig, ax = plt.subplots()
                        ax.plot(data_array)
                        fig.savefig("C:/Users/jakpe/Desktop/Studiet/"
                                    + "PMTV0_afterpulse/errorserror_example_" 
                                    + str(i) + ".pdf")
                        plt.close()
                        error_ite = i
                        print("Error iteration: ", i)
                        continue
                    
                    if i - 1 == error_ite:
                        fig, ax = plt.subplots()
                        ax.plot(data_array)
                        fig.savefig("C:/Users/jakpe/Desktop/Studiet/"
                                    + "PMTV0_afterpulse/errorserror_example_" 
                                    + str(i) + ".pdf")
                        plt.close()
                    
                    # Correcting for missing index in wave
                    diff_len = len(data_array) - len(wave_temp)
                    afterpulse_index = np.argmin(wave_temp) + diff_len
                    
                    
                    # Searching for afterpulses to the left
                    for k in range(len(afterpulses_left)):
                        wave_temp = data_array[signal_index + right_i:(
                                afterpulses_i[k] - afterpulses_left[k]
                        )]
    
                        # If window is to small
                        if len(wave_temp) < 100:
                            continue
                        
                        diff_len_temp = (len(data_array[:right_i])
                        - len(wave_temp))
                        afterpulse_index = np.argmin(wave_temp) + diff_len_temp
                        if afterpulse_index in afterpulses_i:
                            continue
                        else:
                            if data_array[afterpulse_index] > threshold:
                                continue
                            break
                    
                    # Searching for afterpulses to the right
                    for k in range(len(afterpulses_right)):
                        wave_temp = data_array[(
                                afterpulses_i[k] + afterpulses_right[k]):]
    
                        # If window is to small
                        if len(wave_temp) < 50:
                            continue
                        
                        diff_len_temp = len(data_array) - len(wave_temp)
                        afterpulse_index = np.argmin(wave_temp) + diff_len_temp
                        if afterpulse_index in afterpulses_i:
                            continue
                        else:
                            if data_array[afterpulse_index] > threshold:
                                continue
                            break
                    
                    # If afterpulse already exist somehow stop search
                    if afterpulse_index in afterpulses_i:
                        break
                    
                    # If afterpulse is too small dont stop search
                    if data_array[afterpulse_index] > threshold:
                        break
                    
                    # If afterpulse is within alice dead time and some recovering time
                    # look elsewhere
                    if ((afterpulse_index - signal_index)*time_between_points
                        < ALICE_deadtime + 0):
                        break
        
                    
                    a_results = peak_finder(data_array, afterpulse_index)
                    if len(a_results[0]) == 0:
                        print(afterpulse_index)
                        continue
                    
                    afterpulses_dict[l], a_left, a_right = (a_results[0],
                                    a_results[1], a_results[2]) 
                    afterpulses_right.append(a_right)
                    afterpulses_left.append(a_left)
                    afterpulses_i.append(afterpulse_index)
                    
                    # Plotting events with many afterpulses
                    # for illustration
                    show_N_afterpulses = 1
                    if (len(afterpulses_left) > show_N_afterpulses
                    and shown_afterpulses < max_show_afterpulses):
                        shown_afterpulses += 1
                        fig_ex, ax_ex = plt.subplots(figsize=(9,5))
                        a = plt.axes([.32, .22, .5, .35])
                        a.set(facecolor="k"
                              )
                        resolution = 40
                        time_plot = time[afterpulse_index - resolution + 15:
                            afterpulse_index + resolution]
                        a.axis(color='white')
                        a.tick_params(axis='both', which='major', labelsize=10)
                        a.plot(time_plot, data_array[afterpulse_index + 15
                                                     - resolution:
                            afterpulse_index + resolution],
                                   color='white',
                                   linestyle='--', linewidth=0.1
                                   )
                        a.tick_params(axis='x', labelcolor='w')
                        a.tick_params(axis='y', labelcolor='w')
                        time_show = 9400
                        ax_ex.plot(time[time<time_show], 
                                   data_array[time<time_show],
                                   color='white', linewidth=0.05)
                        ax_ex.plot([time[afterpulse_index], 4100],
                                   [data_array[afterpulse_index], min(data_array)/2], 
                                   linewidth=1, color='red', linestyle='--')
                        ax_ex.set(xlabel="Time [ns]",
                                  ylabel="Voltage [V]",
                                  facecolor="k")
                        fig_ex.savefig("figures_events/event_k_r_" 
                                       + str(i) + ".pdf")
                        
            
                # Appending signal and afterpulse events  
                time_now += record_time
                peaks.append(signal)
                peak_class.append(0)
                N_signals += 1
                time_arrival.append(time_now)
                integrals.append(event_integral(signal))
                
                
                for k in range(len(afterpulses_dict)):
                   peaks.append(afterpulses_dict[k])
                   peak_class.append(1)
                   N_afterpulse += 1
                   time_arrival.append(time_now 
                                       + (afterpulses_i[k]
                   - signal_index)*time_between_points)
                   afterpulse_times.append((afterpulses_i[k]
                   - signal_index)*time_between_points)
                   amp_ratio.append(min(afterpulses_dict[k])/
                                    min(signal))
                   amp_afterpulses.append(min(afterpulses_dict[k]))
                   int_afterpulses.append(event_integral(afterpulses_dict[k]))
                   integrals.append(event_integral(afterpulses_dict[k]))
                   


        file_number += 1
        rate.append(N_afterpulse/N_signals)
        
    return (np.array(peaks),
            np.array(peak_class),
            np.array(time_arrival),
            np.array(integrals),
            np.array(afterpulse_times), 
            np.array(amp_ratio),
            np.array(amp_afterpulses),
            np.array(int_afterpulses),
            time_between_points,
            np.array(rate))

if __name__ == "__main__":
# =============================================================================
#   Files and path specification to run through the initial analysis  
# =============================================================================
    PATH0 = "C:/Users/jakpe/Desktop/studiet/PMTV0_afterpulse/"
    
    """Import binary"""
    PATH1 = PATH0 + "data_1900V_50mV/"
    PATH2 = PATH0 + "data_25_06_19_t_medium/"
    PATH3 = PATH0 + "data_25_06_19_t_high/"
    PATH4 = PATH0 + "data_25_06_19_t_verylow/"
    PATH5 = PATH0 + "data_1700V_50mV/"
    PATH6 = PATH0 + "data_1600V_50mV/"
    PATH7 = PATH0 + "data_1500V_50mV/"

    # 1900 V data
    Datafile1 = PATH1 + "1900V5000eventsA.bin"
    Datafile2 = PATH1 + "1900V5000eventsB.bin"
#    Datafile3 = PATH1 + "10000events.bin"
#    Datafile4 = PATH4 + "10000events.bin"
    
    
    # 1700 V data
#    Datafile5 = PATH5 + "1000eventsA.bin"
#    Datafile6 = PATH5 + "7000eventsA.bin"
#    Datafile7 = PATH5 + "7000eventsB.bin"
#    Datafile8 = PATH5 + "1000eventsB.bin"
    Datafile9 = PATH5 + "1700V5000eventsA.bin"
    Datafile10 = PATH5 + "1700V5000eventsB.bin"
    Datafile11 = PATH5 + "1700V5000eventsC.bin"
    Datafile18 = PATH5 + "1700V5000eventsD.bin"
    Datafile19 = PATH5 + "1700V5000eventsE.bin"
    Datafile20 = PATH5 + "1700V5000eventsF.bin"
    
    # 1600 V data
    Datafile17 = PATH6 + "1600V5000eventsA.bin"
    Datafile21 = PATH6 + "1600V5000eventsB.bin"
    
    # 1500 V data
    Datafile12 = PATH7 + "1500V5000eventsA.bin"
    Datafile13 = PATH7 + "1500V5000eventsB.bin"
    Datafile14 = PATH7 + "1500V5000eventsC.bin"
    Datafile15 = PATH7 + "1500V5000eventsD.bin"
    Datafile16 = PATH7 + "1500V5000eventsE.bin"
    
    files = [
             Datafile1, 
             Datafile2,
#             Datafile3,
#             Datafile4,
#             Datafile5,
#             Datafile6,
#             Datafile7,
#             Datafile8,
             Datafile9,
             Datafile10,
             Datafile11,
             Datafile12,
             Datafile13,
             Datafile14,
             Datafile15,
             Datafile16,
             Datafile17,
             Datafile18,
             Datafile19,
             Datafile20,
             Datafile21,
             ]
    
    Events = [5000 for i in files]

    Event_length = GetBinaryLength(files[0], Events[0])
    
    print("Event length:")
    print(Event_length)
    
    try:
        peaks
    except NameError:
        (peaks, peak_class, 
             time_arrival,
             integrals_all,
             afterpulse_times, 
             amp_ratio,
             amp_afterpulses,
             int_afterpulses,
             time_between_points,
             rate)  = (ImportBinary(files, Event_length, Events))
    
    # Getting isolated time and energy diff
    isolated_t = np.zeros(len(time_arrival))
    isolated_e = np.zeros(len(time_arrival))
    for ite in range(len(time_arrival)):
        
        # First peak
        if ite == 0:
            if time_arrival[ite+1] - time_arrival[ite] <= start_time:
                isolated_t[ite] = time_arrival[ite+1] - time_arrival[ite]
                isolated_e[ite] = integrals_all[ite+1]/integrals_all[ite]
            else:
                isolated_t[ite] = start_time + np.random.normal(ALICE_deadtime,
                                                            ALICE_deadtime/5)
                isolated_e[ite] = 6
        
        # Last peak
        elif ite == len(time_arrival) - 1:
            if (time_arrival[ite] - time_arrival[ite - 1] <=
                len(time_arrival)*record_time - time_arrival[ite]):
                isolated_t[ite] = time_arrival[ite] - time_arrival[ite - 1]
                isolated_e[ite] = integrals_all[ite]/integrals_all[ite - 1]
            else:
                isolated_t[ite] = (len(time_arrival)*record_time 
                          - time_arrival[ite])
                isolated_e[ite] = integrals_all[ite]/np.random.normal(
                        np.mean(integrals_all),
                        np.std(integrals_all))
        
        # All other peaks
        else:
            isolated_t[ite] = time_arrival[ite] - time_arrival[ite - 1]
            isolated_e[ite] = integrals_all[ite]/integrals_all[ite - 1]

    
    
# =============================================================================
#   Timing calculation and isolation correction
# =============================================================================
    def time_of_ion_r5924(M_Q):
        """ Based on setup and M/Q will the 
        expected time of arrival be returned"""
        V = 180 # Volts
        constant = 1.134
        L = 0.55 # in cm
        t = constant * np.sqrt(L**2 * M_Q / V) # in microseconds
        return t
        
    # What ions?    
    ions = ["H$^+$", 
            "H$_2^+$",
            "He$^+$", 
            "CH$_4^{++}$", 
            "CH$_4^+$",
            "Ne$^+$",
            "N$_2^+$", 
            "O$_2^+$",
            "Ar$^+$",
            "Xe$^{++}$",
            "Xe$^{+}$", 
            ]    
    ions_M_Q = [1, 
                2.5, 
                4,
                8,
                16,
                20,
                28,
                32,
                40,
                65.5,
                131]
    expected_time = np.zeros(len(ions_M_Q))
    for ite in range(len(ions_M_Q)):
        expected_time[ite] = time_of_ion_r5924(ions_M_Q[ite])
    
    # Correcting for isolation (blindness)
    for ite in range(len(isolated_t)):
        if isolated_t[ite] == 10000:
            
#            r = np.random.normal(start_time,
#                             ALICE_deadtime)
#            while r < start_time:
#                r = np.random.normal(start_time,
#                             ALICE_deadtime)
#            isolated_t[ite] = start_time + r
            isolated_t[ite] = np.random.uniform(ALICE_deadtime, 10000)
            isolated_e[ite] = integrals_all[ite]/np.random.normal(
                        np.mean(int_afterpulses),
                        np.std(int_afterpulses))
    
# =============================================================================
#   Plotting and printing
# =============================================================================
    mpl.rcParams['font.size'] = 16   
    
    # Distribution of times
    figt, axt = plt.subplots(figsize=(13,10))
    time_max = max(expected_time)*1000+300
    hist, bins = np.histogram(afterpulse_times, bins=90, 
                              range=(0,time_max))
    t_var = np.linspace(bins[0], bins[-1], len(hist))
    t_var = t_var[hist>0]
    hist = hist[hist>0]
    f_interpolate = interp1d(t_var, np.log10(hist), kind='cubic')
    x_interpolate = np.linspace(t_var[0], t_var[-1], 10000)
    axt.plot(x_interpolate, f_interpolate(x_interpolate),
                 linestyle='--', label='Cubic interpolation') 
    axt.plot(t_var, np.log10(hist), label='Data',
                 linestyle='', marker='o', markersize=4, linewidth=0.1)
    axt.grid(True, color='black', linestyle='--', linewidth=0.5, alpha=0.25)
    axt.set(xlabel="Time of arrival after signal pulse",
            ylabel="Frequency log scale")
    random_height = np.random.uniform(1,2,len(expected_time))
    for ite in range(len(expected_time)):
        y = max(np.log10(hist))+(1.5*np.sin(4.1*ite+0.5))**2
        axt.plot([expected_time[ite]*1000, expected_time[ite]*1000],
             [0, y],
             linestyle='--', color='red', alpha=0.5,
             linewidth=0.7)
        axt.plot([expected_time[ite]*1000,
                 (expected_time[ite])*1000+len(ions[ite])*time_max/200],
             [y,y],
             linestyle='--', color='red', alpha=0.5,
             linewidth=0.7)
        axt.text(expected_time[ite]*1000, y, ions[ite])
    axt.legend()
    figt.savefig("timedistribution.pdf")
    
    # Distribution of ratios
    figr, axr = plt.subplots(figsize=(13,10))
    rmin, rmax, Nbins = 0.05, 100, 68
    histr, binsr = np.histogram(amp_ratio*100, bins=Nbins, range=(rmin,rmax))
    r_var = np.linspace(min(binsr),max(binsr),len(histr))
    r_var = r_var[histr>0]
    histr = histr[histr>0]
    error = np.array([np.sqrt(i) for i in histr])
    axr.errorbar(r_var, histr, xerr=np.ones(len(histr))*binsr[1], 
                 yerr=error, alpha=0.7,
            color='black', linestyle='--', linewidth=0.3,
            marker='+', markersize=4,
            label='Afterpulse energy ratios')
    axr.grid(True, color='black', linestyle='--', linewidth=0.5, alpha=0.25)
    axr.set(xlabel='Afterpulse energy and corresponding signal energy ratio [%]',
            ylabel='Frequency per ' + str(round(binsr[1],2)) + '%')
    
    # Fit
    minuit_gaussian, xfit, yfit, text_fit = (
            quad_GaussianFit(Nbins, rmin, rmax, amp_ratio*100))
    axr.plot(xfit, yfit, label=text_fit)
    for i in range(1,5):
        axr.plot(xfit, gauss_extended(xfit, minuit_gaussian.values['N'+str(i)],
                                  minuit_gaussian.values['mu'+str(i)],
                                  minuit_gaussian.values['sigma'+str(i)]), 
        label=("Gaussian fit number " + str(i) + " with mean @ " +
               str(round(minuit_gaussian.values['mu'+str(i)],2))) + "%")
    axr.legend()
    figr.savefig("ratiodistribution.pdf")
 
    # Distribution of signals to be printed
    amp_signals = np.array([min(i) for i in peaks[peak_class==0] if i != []])
    amp_low = amp_signals[amp_signals>-0.100]
    amp_medium = amp_signals[amp_signals<-0.100]
    amp_medium = amp_medium[amp_medium>-0.200]
    amp_high = amp_signals[amp_signals<-0.200]
    amp_high = amp_high[amp_high>-0.3]
    amp_vhigh = amp_signals[amp_signals<-0.4]
    amp_vhigh = amp_vhigh[amp_vhigh>-0.5]
    amp_toohigh = amp_signals[amp_signals<-0.5]
    print("Amount of signals in each group of pulse heights")
    print("50-100 mV:", len(amp_low))
    print("100-200 mV:", len(amp_medium))
    print("200-300 mV:", len(amp_high))
    print("300-400 mV:", len(amp_vhigh))
    print("500-max mV:", len(amp_toohigh))
    
    # Removing very energitic events to balance the integral weight
    afterpulses = peaks[peak_class==1]
    afterpulses_iso_t = isolated_t[peak_class==1]
    afterpulses_iso_e = isolated_e[peak_class==1]
    afterpulses = afterpulses[amp_afterpulses > max_signal_amplitude]
    afterpulses_iso_t = afterpulses_iso_t[amp_afterpulses > max_signal_amplitude]
    afterpulses_iso_e = afterpulses_iso_e[amp_afterpulses > max_signal_amplitude]
    
    signals = peaks[peak_class==0]
    signals_iso_t = isolated_t[peak_class==0]
    signals_iso_e = isolated_e[peak_class==0]
    signals = signals[amp_signals > max_signal_amplitude]
    signals_iso_t = signals_iso_t[amp_signals > max_signal_amplitude]
    signals_iso_e = signals_iso_e[amp_signals > max_signal_amplitude]
    
# =============================================================================
#   Export for further analysis
# =============================================================================
    
    pd.DataFrame({'a':afterpulses,
                  'isolated_time':afterpulses_iso_t,
                  'energy_difference':afterpulses_iso_e}
                   ).to_csv("afterpulses.csv", index=False)
    pd.DataFrame({'s':signals,
                  'isolated_time':signals_iso_t,
                  'energy_difference':signals_iso_e}
                   ).to_csv("signals.csv", index=False)
    pd.DataFrame(np.array([time_between_points])).to_csv("time_between_points.csv")
    
    print()
    print("Number of each class:")
    print("Signals (0):", len(signals))
    print("Afterpulses (1):", len(afterpulses))