# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:38:39 2014

@author: s362khan

    CONDUCTANCE MODEL - 2D
    
"""
import numpy as np
import matplotlib.pyplot as plt

def main(s_angleDeg, t_stop = 6000):
    ''' ---------------------------------------------------------
    INPUT
    -----------------------------------------------------------'''
    
    ''' Stimulus '''
    s_angleRad = s_angleDeg * np.pi / 180
    s = np.array([np.cos(s_angleRad), np.sin(s_angleRad)])
    
    print("Input stimulus: s= [%f, %f], Angle %f, Input = %i" \
    %(s[0], s[1], np.arctan2(s[1], s[0])*180/np.pi, s_angleDeg) )
    
    # Setup Direction Vector Figure and plot stimulus
#    plt.figure("2D space", figsize = (8,8))
#    plt.plot([0, s[0]], [0, s[1]], color = "red", linewidth = 2, \
#             label = 'stimulus')
#    plt.xlim([-1.1, 1.1])
#    plt.ylim([-1.1, 1.1])
#    fig1Ax = plt.gca()
#    fig1Ax.spines['right'].set_color('none')
#    fig1Ax.spines['top'].set_color('none')
#    fig1Ax.xaxis.set_ticks_position('bottom')
#    fig1Ax.spines['bottom'].set_position(('data', 0))
#    fig1Ax.yaxis.set_ticks_position('left')
#    fig1Ax.spines['left'].set_position(('data', 0))
#    fig1Ax.set_title("Preferred Stimulus Angle %i" \
#                     %(np.arctan2(s[1], s[0]) * 180 / np.pi) )
                     
    ''' ------------------------------------------------------\
    PRESYNAPTIC NEURON POPULATION
    --------------------------------------------------------'''
    pre_n = 100
    pre_rMax = 100   
    
    ''' Preferred Direction Vectors Distribution '''
    # Uniform Distribution
    pre_preferredStimAngle = np.linspace(0, 2*np.pi, pre_n, endpoint = False)
    
    # Random Distribution
#    pre_preferredStimAngle = np.random.uniform(0, 2*np.pi, pre_n) 

    # Cricket Cercal System [given preferred Direction Vector Distribution]
#    pre_n = 4
#    pre_preferredStimAngle = np.array([45, 135, -45 , -135]) * np.pi / 180

    # 2 = Dimensions of stimulus
    pre_preferredStim = np.zeros(shape = (pre_n, 2))
    pre_preferredStim[:, 0] = np.cos(pre_preferredStimAngle)
    pre_preferredStim[:, 1] = np.sin(pre_preferredStimAngle)
    
#    plt.figure("2D space")    
#    for x, y in pre_preferredStim: 
#        plt.plot([0, x], [0, y], color = "blue", linewidth = 0.5, \
#        linestyle = '--', label = 'preferred stimulus')

    ''' Firing Rates '''
#    # Cosine Tuning Curves
#    pre_fireRates = np.dot(s, pre_preferredStim.T) * pre_rMax
#    pre_fireRates[pre_fireRates < 0] = 0
    
    # Gaussian Tuning Curves
    pre_var = 0.5  # width of tuning curve
    
    # Assume unit length preferred direction vectors
    pre_angleStimPreferred = np.arccos( \
                            np.dot(pre_preferredStim, s) / np.linalg.norm(s))
    
    pre_fireRates = pre_rMax * \
        np.exp(-np.square(pre_angleStimPreferred / pre_var) / 2)

    # Plot Firing Rates
#    plt.figure("Firing Rates")
#    plt.title("Firing Rate", y = 1.05)
#    plt.stem(np.arange(0, pre_n), pre_fireRates, label = 'preSyp Out')
#    fig2Ax = plt.gca()
#    fig2Ax.set_xlabel('Neuron Index', fontsize = 9, x = 1)
#    fig2Ax.set_xlim([0, (1.1 * pre_n)])
#    fig2Ax.set_ylabel('Firing Rate', fontsize = 9)
#    fig2Ax.set_ylim([0, (1.1 * pre_fireRates.max())])
    
    ''' Spike Generation  '''
    # Poisson Firing - Commulative expontential inter-spike times method
    pre_spikeTimes = list()
    
    for rate in pre_fireRates:
        spikeTimes = []
        
        if rate != 0 and not np.isnan(rate):
            nSpikes = np.floor(2 * t_stop * rate / 1000.0)
            spikeTimes = np.cumsum( \
            np.random.exponential(scale = 1000.0/rate, size = nSpikes) )
        
        pre_spikeTimes.append(spikeTimes)

    # Convert to indices and eliminate invalid entries
    for row in np.arange(len(pre_spikeTimes)):
        pre_spikeTimes[row] = \
        [ int(x) for x in pre_spikeTimes[row] if x < t_stop ]
    
    # Print number of spikes for each neuron
#    for idx, item in enumerate(pre_spikeTimes):
#        print ("preferred Stimulus %0.4f, Rate %i" %((pre_preferredStim[idx]*180/np.pi), \
#        len(item)))
    
    pre_out = np.zeros(shape =( pre_n, t_stop))
    for row in np.arange(len(pre_spikeTimes)):
        pre_out[row, pre_spikeTimes[row]] = 1 
    
     # Plot the Spikes of each neuron over Time
#    fig3 = plt.figure("PostSynaptic Input Spikes")
#    plt.title("PostSynaptic Input Spikes")
#    plt.ylabel('Neuron Index')
#    if pre_n <= 25:
#        plt.yticks(np.arange(pre_n))
#    plt.xlabel('Time(ms)')
#    im = plt.imshow(pre_out, aspect = 'auto', cmap = plt.cm.binary)
    
    ''' -------------------------------------------------------------
    POSTSYNAPTIC NEURON
    --------------------------------------------------------------'''

    ''' Fire Rates Estimation '''
    # Average incoming spikes using a window to get instantaneous rate
    # Alpha function Window = alpha^2 * x * exp(-alpha * x)
    post_alpha = 1/16.
    post_windowLen = 100
    
    tmp =  np.linspace(0, post_windowLen-1, post_windowLen)
    post_window = np.power(post_alpha, 2) * tmp * np.exp(-post_alpha * tmp)
    
#    # Plot spike rate estimatation window
#    plt.figure("Firing Rate Estimation Averaging Window")
#    plt.plot(tmp, post_window)
#    plt.title("Alpha Function Window (alpha = %f)" %post_alpha)
#    plt.xlabel('Time (ms)', fontsize = 9, x = 1)
#    plt.ylabel('Weight', fontsize = 9)
    
    # Convolve spikes with window to get instantaneous fire rate
    post_instRates = np.zeros(shape = pre_out.shape)
    for time in np.arange(t_stop):
        for synapse in np.arange(pre_n):
            spikes = pre_out[synapse, time::-1]
            
            if len(spikes) > post_windowLen:
                spikes = spikes[0:post_windowLen]
           
            post_instRates[synapse, time] = \
            np.convolve(spikes, post_window).sum()
    
    # Override instantaneous firing rates with actual firing rates
#    post_instRates = np.repeat(pre_fireRates[:,np.newaxis], repeats = t_stop, axis = 1)

    ''' Conductance Model '''
    
    # Reversal Potential (mV)
    post_XE =   0.0 * np.power(10.0, -3)
    post_IE = -70.0 * np.power(10.0, -3)
    
    # Voltage Thresholds
    post_thReset = -70.0 * np.power(10.0, -3)
    post_thAp    = -50.0 * np.power(10.0, -3)
    post_vAp     =  50.0 * np.power(10.0, -3)  # Peak Action Potential voltage
    
    # Specific Capacitance
    post_cm = 10 * np.power(10.0, 3) 
    
    post_v = np.zeros(shape = (t_stop, 1))
    post_rateCalc = np.zeros(shape = (t_stop, 1))
    
    #Membran Potential
    t_lastSpike = 0
    post_nSpikes = 0
    
    for time in np.arange(t_stop):
        rates = post_instRates[:, time]
        timeLastSpike = time - t_lastSpike
        
        XconRad = (rates * pre_preferredStim.T).sum(axis = 1)
        Xcon = np.arctan2(XconRad[1],XconRad[0]) * 180 / np.pi
        Xcon += 180 # Add a DC value to make all firing rates positive
        
        #limit Range of Conductance
        if Xcon >= 360:
            Xcon = 360
        elif Xcon <= 0:
            Xcon = 0
        
        Icon = rates.sum()
        totalCon = Xcon + Icon
        
        vInf = (Xcon * post_XE + Icon * post_IE) / totalCon
        vK = (Xcon * (post_thReset - post_XE) + Icon * (post_thReset - post_IE)) / totalCon
        vExp = np.exp(-totalCon * timeLastSpike / post_cm)
        
        post_v[time] = vInf + vK*vExp
        
#        print( "time=%i, timeLastSpike=%i, Xcon=%f, Icon=%f, vInf=%f, vK=%f, vExp=%f" \
#            %(time, timeLastSpike, Xcon, Icon, vInf, vK, vExp) )
            
        if post_v[time] >= post_thAp:
            post_v[time] = post_vAp
            t_lastSpike = time
            post_nSpikes += 1.0
            #raw_input()        

        # Firing Rate
        # Theoretical Firing Rate
        post_tIsi = np.log( (post_thAp*totalCon - Xcon*post_XE - Icon*post_IE ) / \
                        (post_thReset*totalCon - Xcon*post_XE - Icon*post_IE ) ) \
                *(-post_cm / totalCon)
                     
        post_rateCalc[time] = 1 / post_tIsi

    # Measured Firing Rate - Simple Binning
    post_rateMeas = post_nSpikes / t_stop
    post_rateCalcMean = np.mean(post_rateCalc)
    post_rateCalcVar = np.var(post_rateCalc)

#    # Plot post-synaptic Neuron Voltage
#    plt.figure("Post Synaptic Membrane Potential")
#    plt.title("Post Synaptic Membrane Potential, Stimulus %i" %s_angleDeg)
#    plt.plot(np.arange(t_stop), post_v * np.power(10.0,3))
#    plt.plot(np.arange(t_stop), post_thAp * np.ones(shape = (t_stop, 1)) * np.power(10.0,3), 'r', label = 'AP Thresh')
#    ax = plt.gca()
#    ax.set_xlabel('Time(ms)', x = 1)
#    ax.set_ylabel("Membrane Potential(mV)")
#    ax.legend()       
#        
#    plt.figure("Post Synaptic Firing Rates")
#    plt.plot(np.arange(t_stop), post_rateCalc, label='Calculated Rate')
#    ax = plt.gca()
#    ax.set_xlabel('Time(ms)', x = 1)
#    ax.set_ylabel("Firing Rate")
#    plt.axhline(y=post_rateMeas, color='red', linewidth=2, label='Measured Rate')
#    plt.legend()
    
    return(post_rateMeas, post_rateCalcMean, post_rateCalcVar)


if __name__ == "__main__":

    #pyplot interactive mode - do not need to close windows after each run
    plt.ion()

    s_angle = np.linspace(-180, 180, num = 360, endpoint = False)
    #s_angle = [ -150]
    
    # Simulation Time per stimulus 
    stepDuration = 1000
    print ("Simulation time(ms) per Stimulus %i" %stepDuration)
    
    measRate = np.zeros(shape = len(s_angle))
    calcRate = np.zeros(shape = len(s_angle))
    calcRateVar = np.zeros(shape = len(s_angle))
    
    for idx, stimulus in enumerate(s_angle):
        measRate[idx], calcRate[idx], calcRateVar[idx] = main(stimulus, stepDuration)
    
    plt.figure("Post Synaptic Neuron")
    plt.title("PostSynaptic Neuron Tuning Curve (Conductance Model)")
    plt.plot(s_angle, measRate, label = 'Mean Firing Rate (measured)')
    plt.plot(s_angle, calcRate, 'r--', label = 'Mean FiringRate (calculated)' )
    ax = plt.gca()
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Firing Rate')
    plt.legend()
    
    # Plot Variance Vertical Lines
    for idx, stimulus in enumerate(s_angle):
        plt.plot([stimulus, stimulus], [(calcRate[idx] + calcRateVar[idx]), \
                                        (calcRate[idx] - calcRateVar[idx])], 'k')
    
    plt.figure('Post Synaptic Response Variance')
    ax = plt.gca()
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Variance')
    plt.plot(s_angle, calcRateVar)
    plt.title('Post Synaptic Firing Rate Variance')
    
    
