# -*- coding: utf-8 -*-
"""--------------------------------------------------------------------------------------
Created on Thu May 29 18:24:20 2014

@author: salman Khan

     MAXIMUM LIKLIEHOOD DECODER

--------------------------------------------------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt


def main(s_angleDeg):
    ''' ---------------------------------------------------------
    INPUT
    -----------------------------------------------------------'''
    
    ''' Stimulus '''
    s_angleRad = s_angleDeg * np.pi / 180
    s = np.array([np.cos(s_angleRad), np.sin(s_angleRad)])
    
    # Setup Direction Vector Figure and plot stimulus
    plt.figure("2D space", figsize = (8,8))
    plt.plot([0, s[0]], [0, s[1]], color = "red", linewidth = 2, \
             label = 'stimulus')
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    fig1Ax = plt.gca()
    fig1Ax.spines['right'].set_color('none')
    fig1Ax.spines['top'].set_color('none')
    fig1Ax.xaxis.set_ticks_position('bottom')
    fig1Ax.spines['bottom'].set_position(('data', 0))
    fig1Ax.yaxis.set_ticks_position('left')
    fig1Ax.spines['left'].set_position(('data', 0))
    fig1Ax.set_title("Preferred Stimulus Angle %i" \
                     %(np.arctan2(s[1], s[0]) * 180 / np.pi) )
    
    # Time
    t_stop = 6000 # time im ms
    
    ''' ------------------------------------------------------\
    PRESYNAPTIC NEURON POPULATION
    --------------------------------------------------------'''
    pre_n = 100
    pre_rMax = 100      # per second
    
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
    for x, y in pre_preferredStim: 
        plt.plot([0, x], [0, y], color = "blue", linewidth = 0.5, \
        linestyle = '--', label = 'preferred stimulus')
        
    ''' Firing Rates '''
    # Cosine Tuning Curves
    pre_fireRates = np.dot(s, pre_preferredStim.T) * pre_rMax
    pre_fireRates[pre_fireRates < 0] = 0
    
    # Gaussian Tuning Curves
#    pre_var = 1  # width of tuning curve
#    
#    # Assume unit length preferred direction vectors
#    pre_angleStimPreferred = np.arccos( \
#                            np.dot(pre_preferredStim, s) / np.linalg.norm(s))
#    
#    pre_fireRates = pre_rMax * \
#        np.exp(-np.square(pre_angleStimPreferred / pre_var) / 2)
    
    # Plot Firing Rates
    plt.figure("Firing Rates")
    plt.title("Firing Rate", y = 1.05)
    plt.stem(np.arange(0, pre_n), pre_fireRates, label = 'preSyp Out')
    fig2Ax = plt.gca()
    fig2Ax.set_xlabel('Neuron Index', fontsize = 9, x = 1)
    fig2Ax.set_xlim([0, (1.1 * pre_n)])
    fig2Ax.set_ylabel('Firing Rate', fontsize = 9)
    fig2Ax.set_ylim([0, (1.1 * pre_fireRates.max())])
    
    ''' Spike Generation  '''
    # Poisson Firing - Commulative expontential inter-spike times method
    pre_spikeTimes = list()
    
    for rate in pre_fireRates:
        spikeTimes = []
        
        if rate != 0:
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
#        print "preferred Stimulus:", pre_preferredStimAngle[idx]*180/np.pi, \
#        "Rate: ",len(item)

    pre_out = np.zeros(shape =( pre_n, t_stop))
    for row in np.arange(len(pre_spikeTimes)):
        pre_out[row, pre_spikeTimes[row]] = 1 
    
    # Plot the Spikes of each neuron over Time
    fig3 = plt.figure("PostSynaptic Input Spikes")
    plt.title("PostSynaptic Input Spikes")
    
    plt.ylabel('Neuron Index')
    if pre_n <= 25:
        plt.yticks(np.arange(pre_n))

    plt.xlabel('Time(ms)')
    im = plt.imshow(pre_out, aspect = 'auto', cmap = plt.cm.binary)
#    fig3.colorbar(im)    
    
    ''' ------------------------------------------------------\
    POSTSYNAPTIC NEURON
    --------------------------------------------------------'''

    ''' Instantaneous Rate Estimation '''
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
#    post_instRates = np.repeat(pre_fireRates[:,np.newaxis], repeats = t_stop,
#                               axis = 1)
    
    ''' 
    Maximum Likelihood Decoding ----------------------------------
    '''
    post_sEstML = np.zeros(shape = (t_stop, 2))
    
    for time in np.arange(t_stop):
        rates = post_instRates[:, time]
        if rates.sum() != 0:
            post_sEstML[time] = (rates *pre_preferredStim.T).sum(axis = 1) / rates.sum()

#    # Plot Stimulus Estimates
#    plt.figure("2D space")
#    for x,y in post_sEstML:
#        plt.plot([0,x],[0,y], 'm') 
    
    ''' Error Estimation '''
    post_StimEstDot = np.dot(post_sEstML, s)
    post_EstMag = [np.linalg.norm(estimate) for estimate in post_sEstML]
    post_StimMag = np.linalg.norm(s)
    
    post_AngleError = (180/np.pi) * np.arccos( \
        post_StimEstDot / (post_EstMag * post_StimMag) )
    
    # Plot Estimate Error
    plt.figure("Error Estimate")
    plt.plot(np.arange(t_stop), post_AngleError)

    post_ErrorAvg = np.mean(post_AngleError)
    print "Stimulus %f, Average Error %f" %( s_angleDeg, post_ErrorAvg)
    
    return (post_ErrorAvg)

if __name__ == "__main__":
    # if you call this script from the command line (the shell) it will
    # run the 'main' function
        
    #pyplot interactive mode - do not need to close windows after each run
    plt.ion()
    
    s_angle = np.linspace(-180, 180, num = 360, endpoint = False)
    s_angle =[45]
    
    Error = np.zeros(shape = (len(s_angle), 1))
    
    for idx, stimulus in enumerate(s_angle):
        Error[idx] = main(stimulus)
        
    plt.figure("Avg Error")
    plt.plot(s_angle, Error, '+r-')
    
