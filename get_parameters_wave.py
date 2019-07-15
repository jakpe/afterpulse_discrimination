# -*- coding: utf-8 -*-
"""
@author: jakpe
"""
import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
import matplotlib as mpl
mpl.rcParams['font.size'] = 15                         # Set the general plotting font size
from sklearn.utils import shuffle
import pandas as pd
import iminuit
from probfit import Chi2Regression
import seaborn as sns


Unit = 10.**(-9) #ns
global time_between_points
time_between_points = float(list(
        pd.read_csv("time_between_points.csv")["0"])[0])

global upper_limit, lower_limit
upper_limit = 0.9
lower_limit = 0.1

global threshold
threshold = -0.149


def event_integral(wave):
    """ Get integral under peak above threshold, 
    and since this is a negative polarized PMT,
    a minus sign is put in front """
    return - sum(wave[wave < min(wave)*0.9])

def second_degree(x,a,b,c):
    """ 2nd degree poly function """
    return a*x**2 + b*x + c

def _FWHM(rising, top, falling):
    """ Full width half max (FWHM) """
    wave = np.append(rising, top)
    wave = np.append(wave, falling)
    
    fwhm = 2*np.sqrt(2*np.log(2))*np.std(wave)
    return fwhm

def fit_2nd(voltage, time, vol_time1, vol_time2=0, plot=False, rise=True): 
    """ Fit 2nd degree polynomial and return time at 
    a certain point or difference in time between two
    points (For fall and rise time)"""
    a, b, c = np.polyfit(time, voltage, 2)
    
    point_of_interesting = c - vol_time1
 
    t1 = solve_for_time(a, b, c, point_of_interesting,rise)
    t2 = 0
    
    if vol_time2 != 0:
        point_of_interesting2 = c - vol_time2
        t2 = solve_for_time(a, b, c, point_of_interesting2, rise)
    
    if plot:
        fig, ax = plt.subplots()
        time_var = np.linspace(min(time),max(time),10000)
        ax.plot(time, voltage, 'o', label='data')
        ax.plot(time_var, second_degree(time_var, a, b, c),
                label='2nd degree poly fit')
        ax.legend()

    solution = t1 - t2

    return a, b, c, solution
    
def solve_for_time(a, b, c, point_of_interesting, rise):
    """ solve 2nd poly fit to find specific time """
    
    d = b**2 - 4*a*point_of_interesting

    if d < 0:
        solution1, solution2 = -b/(2*a), -b/(2*a)
    else:
        solution1, solution2 = (-b + np.sqrt(d))/(2*a), (-b - np.sqrt(d))/(2*a)
    
    if rise:
        if a < 0:
            if solution2 > solution1:
                solution1 = solution2
        else:
            if solution2 < solution1:
                solution1 = solution2
    else:
        if a < 0:
            if solution2 < solution1:
                solution1 = solution2
        else:
            if solution2 > solution1:
                solution1 = solution2
    return solution1

def constant_check(top,t_top):
    """ Reduced chi-square of top being constant """
    def constant_func(x,a):
        return a
    top, t_top = np.array(top), np.array(t_top)
    
    error = np.array([np.sqrt((i - np.mean(top))**2) for i in top])
    chi2_object = Chi2Regression(constant_func, t_top, top, weights=error)
    minuit = iminuit.Minuit(chi2_object, pedantic=False, a=0, print_level=0)
    minuit.migrad();   
    NDOF = len(top) - 1
    Chi2_fit = minuit.fval
    
    if NDOF == 0:
        NDOF = 1
    
    return Chi2_fit/NDOF

def oscillation_counter(sidewave, sidewave_time, a, b, c):
    """ Takes on side of the wave and
    the corresponding time and define a 
    line based on the 2nd degree poly. 
    Then it counts how many 
    times that line is crosse by scanning 
    through all the points """
    counter = 0
    
    for ite in range(len(sidewave) - 1):
        if ((sidewave[ite] > second_degree(sidewave_time[ite], a, b, c)
        and sidewave[ite+1] < second_degree(sidewave_time[ite+1], a, b, c)) or
            (sidewave[ite] < second_degree(sidewave_time[ite], a, b, c)
        and sidewave[ite+1] > second_degree(sidewave_time[ite+1], a, b, c))):
            counter += 1
            
    return counter

def top_time(top):
    """ Top time """
    return len(top)*time_between_points

def total_time(rise_time, fall_time, top_time):
    """ Total time or (FWTM)"""
    return rise_time + fall_time + top_time

def splitting(wave, amp):
    """ Wave splitting and time assigning """
    time = np.array(range(len(wave)))*time_between_points
    
    # Top is above 90%
    t_top = time[wave<amp*upper_limit]
    top = wave[wave<amp*upper_limit]
    
    # Rising side is left side of pulse between 10-90% (maybe)
    rising = wave[:np.argmin(wave)]
    t_rising = time[:np.argmin(wave)]
    t_rising = t_rising[rising<amp*lower_limit]
    rising = rising[rising<amp*lower_limit]
    t_rising = t_rising[rising>amp*upper_limit]
    rising = rising[rising>amp*upper_limit]
    
    t_rising = np.append(max(t_rising[rising == max(rising)]),
            t_rising[rising < max(rising)])
    rising = np.append(max(rising),rising[rising < max(rising)])
     
    # Falling side is right side of pulse between 10-90% (maybe)
    falling = wave[np.argmin(wave):]
    t_falling = time[np.argmin(wave):]
    t_falling = t_falling[falling<amp*lower_limit]
    falling = falling[falling<amp*lower_limit]
    t_falling = t_falling[falling>amp*upper_limit]
    falling = falling[falling>amp*upper_limit]
    
    if not len(falling) == 0:
        t_falling = t_falling[:np.argmax(falling) + 1]
        falling = falling[:np.argmax(falling) + 1]
    
    return top, rising, falling, t_top, t_rising, t_falling

# Reserving parameters
a, b, c, d, e, f = [], [], [], [], [], []
total_t, top_t, rise_t, fall_t = [], [], [], []
integral, FWHM, chi2, amp = [], [], [], []
PHR, osc_rising, osc_falling, osc_top = [], [], [], [] 
classification = []

file_A = "afterpulses.csv"
dataframe_afterpulse = pd.read_csv(file_A)

file_S = "signals.csv"
dataframe_signal = pd.read_csv(file_S)

# =============================================================================
# Signals    
# =============================================================================

# In order to the models not to be fed with only signals, a limit is sat here,
# Since the signals are overrepresented in the data
N_group = 10
signals_all = np.array(dataframe_signal['s'])
time_isolated_potential = np.array(dataframe_signal["isolated_time"])
energy_diff_potential = np.array(dataframe_signal["energy_difference"])
signals_all, time_isolateds, energy_diffs = shuffle(signals_all, 
                                      time_isolated_potential, 
                                      energy_diff_potential,
                                      random_state=42)

time_isolateds = []
energy_diffs = []

groups = {}
for i in range(N_group):
    groups[str(i)] = 0

for i in range(len(signals_all)):
    
    # Plot the i'th wave
    if i == 0:
        plot = True
    else:
        plot = False
    
    # Analysis
    wave = []
    list_wave = signals_all[i]
    list_wave = list_wave.split(' ')
    for string in list_wave:
        if string == '' or string == '[':
            continue
        string_temp = string.split('[')
        if  len(string_temp) > 1:
            string_temp = string_temp[1]
        else:
            string_temp = string_temp[0]
        string_temp = string_temp.split(']')[0]
        if string_temp == '':
                    continue
        wave.append(float(string_temp))
        
    wave = np.array(wave)
    if len(wave) == 0:
        continue
    amp_i = min(wave)
    
    
 
    # Appending all parameters
    top, rising, falling, t_top, t_rising, t_falling = splitting(wave, amp_i)
    
    if len(falling) == 0:
        # error if afterpulse is on the edge of record time
        continue
    
    a_i, b_i, c_i, rise_t_i = fit_2nd(rising, t_rising, upper_limit*amp_i, 
                                vol_time2=lower_limit*amp_i, 
                                plot=plot, rise=True)
    e_i, f_i, g_i, fall_t_i = fit_2nd(falling, t_falling, lower_limit*amp_i, 
                                vol_time2=upper_limit*amp_i, 
                                plot=plot, rise=False)
    top_t_i = top_time(t_top)
    
    FWHM.append(_FWHM(rising, top, falling))
    integral.append(event_integral(wave))
    
    osc_rising.append(oscillation_counter(rising, t_rising, a_i, b_i, c_i))
    osc_falling.append(oscillation_counter(falling, t_falling, e_i, f_i, g_i))
    osc_top.append(oscillation_counter(top, t_top, 0, 0, np.mean(top)))
    chi2.append(constant_check(top, t_top))
    amp.append(amp_i)
    
    rise_t.append(rise_t_i); fall_t.append(fall_t_i); top_t.append(top_t_i)
    total_t.append(total_time(rise_t_i, fall_t_i, top_t_i))
    time_isolateds.append(time_isolated_potential[i])
    energy_diffs.append(energy_diff_potential[i])
    a.append(a_i); b.append(b_i); c.append(c_i); 
    d.append(e_i); e.append(f_i); f.append(g_i);
    PHR.append(abs(_FWHM(rising, top, falling)/amp_i))
    classification.append(0)
    

time_isolateds = np.array(time_isolateds)
energy_diffs = np.array(energy_diffs)
time_isolateda = np.array(dataframe_afterpulse["isolated_time"])
energy_diffa = np.array(dataframe_afterpulse["energy_difference"])
time_isolated = np.append(time_isolateds, time_isolateda)
energy_diff = np.append(energy_diffs, energy_diffa)

# =============================================================================
# Afterpulse
# =============================================================================
for i in range(len(dataframe_afterpulse)):
    
    # Plot the i'th wave
    if i == 0:
        plot = True
    else:
        plot = False
    
    # Analysis
    wave = []
    list_wave = dataframe_afterpulse['a'][i]
    list_wave = list_wave.split(' ')
    for string in list_wave:
        if string == '' or string == '[':
            continue
        string_temp = string.split('[')
        if  len(string_temp) > 1:
            string_temp = string_temp[1]
        else:
            string_temp = string_temp[0]
        string_temp = string_temp.split(']')[0]
        if string_temp == '':
            continue
        wave.append(float(string_temp))
            

            
    wave = np.array(wave)
    if len(wave) == 0:
        continue
    amp_i = min(wave)
    
    # Appending all parameters
    top, rising, falling, t_top, t_rising, t_falling = splitting(wave, amp_i)
    
    if len(falling) == 0:
        # error if afterpulse is on the edge of record time
        error_i = i
        time_isolated = np.delete(time_isolated, len(time_isolateds) + i)
        energy_diff = np.delete(energy_diff, len(energy_diffs) + i)
        continue
    
    a_i, b_i, c_i, rise_t_i = fit_2nd(rising, t_rising, upper_limit*amp_i, 
                                vol_time2=lower_limit*amp_i, 
                                plot=plot, rise=True)
    e_i, f_i, g_i, fall_t_i = fit_2nd(falling, t_falling, lower_limit*amp_i, 
                                vol_time2=upper_limit*amp_i, 
                                plot=plot, rise=False)
    top_t_i = top_time(t_top)
    
    FWHM.append(_FWHM(rising, top, falling))
    integral.append(event_integral(wave))
    
    osc_rising.append(oscillation_counter(rising, t_rising, a_i, b_i, c_i))
    osc_falling.append(oscillation_counter(falling, t_falling, e_i, f_i, g_i))
    osc_top.append(oscillation_counter(top, t_top, 0, 0, np.mean(top)))
    chi2.append(constant_check(top, t_top))
    amp.append(amp_i)
    
    rise_t.append(rise_t_i); fall_t.append(fall_t_i); top_t.append(top_t_i)
    total_t.append(total_time(rise_t_i, fall_t_i, top_t_i))
    a.append(a_i); b.append(b_i); c.append(c_i); 
    d.append(e_i); e.append(f_i); f.append(g_i);
    PHR.append(abs(_FWHM(rising, top, falling)/amp_i))
    classification.append(1)

# Label name
label_name = []
for ite in range(len(classification)):
    if classification[ite] == 0:
        label_name.append('Signal')
    else:
        label_name.append('Afterpulse')

# Saving needed parameters
Dataframe = pd.DataFrame({
                         #'amplitude':amp,
                         'Integral':integral,
                         'PHR':PHR,
                         'Poly parameter d':d,
                         'Poly parameter e':e,
                         'Poly parameter f':f,
                         #'FWHM':FWHM,
                         'Chi2-value of top':chi2,
                         'Fall time':fall_t,
                         'Rise time':rise_t,
                         #'Total_time':total_t,
                         #'Oscillation of top':osc_top,
                         'Poly parameter a':a,
                         'Poly parameter b':b,
                         'Poly parameter c':c,
                         'Osc of rising side':osc_rising,
                         'Osc of falling side':osc_falling,
                         'Isolation in time':time_isolated,
                         'Energy ratio':energy_diff,
                         'label':classification, # 0 for signal and 1 for after 
                         })

Dataframe.to_csv('parameters_w_label.csv', index=False)

# Linear corelation between all the parameters
correlation = Dataframe.corr(method='pearson')
mpl.rcParams['font.size'] = 11    
fig, ax = plt.subplots(figsize=(12,10))
g = sns.heatmap(correlation, cmap="coolwarm", annot=True,
                center=0, square=True, #mask=mask,
                vmin=-1, vmax=1, fmt='.3f')
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=330,
    horizontalalignment='right'
);
ax.xaxis.tick_top()
plt.tight_layout()
fig.savefig("correlation_matrix.pdf")
mpl.rcParams['font.size'] = 15    

# For seaborn to plot
Dataframe_plot = pd.DataFrame({
                  'Integral (mV)':integral,
                  'PHR (ns/mV)':PHR,
                  #r'$\chi^2$-value of top':chi2,
                  #'Energy difference [mv]':energy_diff,
                  'Fall time (ns)':fall_t,
                  'Oscillations of rising side':osc_rising,
                  'Isolation in time (ns)':time_isolated,
                  'Class':label_name,
                  })
g = sns.pairplot(Dataframe_plot, hue="Class", 
                 markers=["+", "o"], plot_kws=dict(s=20, edgecolor='none'))
g.savefig('correlation.png',dpi=700)
    
# After evaluating feature importance save most contributing features
Dataframe_opt =  pd.DataFrame({
                 'Integral':integral,
                 'Fall time':fall_t,
                 #'Poly parameter d':d,
                 'Poly parameter e':e,
                 'Poly parameter f':f,
                 #'FWHM':FWHM,
                 'PHR':PHR,
                 'Chi2-value of top':chi2,
                 #'Rise time':rise_t,
                 #'Total_time':total_t,
                 #'Oscillation of top':osc_top,
                 #'Poly parameter a':a,
                 'Poly parameter b':b,
                 #'Poly parameter c':c,
                 'Osc of rising side':osc_rising,
                 #'Osc of falling side':osc_falling,
                 'Isolation in time':time_isolated,
                 'Energy ratio':energy_diff,
                 'label':classification, # 0 for signal and 1 for after 
                  })   
Dataframe_opt.to_csv('parameters_w_label_opt.csv', index=False)
    


