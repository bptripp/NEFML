"""--------------------------------------------------------------------------------------
Created on Thu May 29 18:24:20 2014

@author: salman Khan

     MAXIMUM LIKLIEHOOD DECODER 1 Dimensional

--------------------------------------------------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    ''' -------------------------------------------------------------
    INPUT
    --------------------------------------------------------------'''
    
    ''' Stimulus '''
    s = 3
    
    s_range = 100
    
    print ("Input Stimulus %i" %(s))
    
    
    ''' Time '''
    t_stop = 6000 # time im ms
    
    ''' -------------------------------------------------------------
    PRESYNAPTIC NEURON POPULATION
    --------------------------------------------------------------'''
    pre_n       = 100
    pre_rMax    = 100      # per second
    
    ''' Preferred Direction Vectors Distribution '''
    # Uniform Distribution
    pre_preferredStim = np.linspace(0, s_range, pre_n, endpoint = False)
    
    ''' Firing Rates '''
    # Gaussian Tuning Curves
    pre_var = s_range * 0.1  # width of tuning curve
    pre_fireRates = pre_rMax * np.exp( -((s - pre_preferredStim) / pre_var)**2 / 2 )
    
    # Plot Firing Rates
#    plt.figure("Firing Rates")
#    plt.title("Firing Rate", y = 1.05)
#    plt.stem(np.arange(0, pre_n), pre_fireRates, label = 'pre_fireRates')
#    fig2Ax = plt.gca()
#    fig2Ax.set_xlabel('Neuron Index', fontsize = 9, x = 1)
#    fig2Ax.set_xlim([0, (1.1 * pre_n)])
#    fig2Ax.set_ylabel('Firing Rate', fontsize = 9)
#    fig2Ax.set_ylim([0, (1.1 * pre_fireRates.max())])
#    fig2Ax.legend()
    
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

    ''' Maximum Likelihood Decoding '''
    post_sEstML = np.zeros(shape = (t_stop, 1))
    
    for time in np.arange(t_stop):
        rates = post_instRates[:, time]
        if rates.sum() != 0:
            post_sEstML[time] = (rates * pre_preferredStim.T).sum() / rates.sum()
#
#    # Plot Stimulus Estimates
#    plt.figure("Stimulus Estimates")
#    plt.title("Stimulus Estimates")
#    plt.plot (np.arange(t_stop), s*np.ones(shape = (t_stop, 1)), 'r', label = 'stimulus')
#    plt.plot( np.arange(t_stop), post_sEstML,  label = 'Estimates')
    
    ''' Conductance Model '''
 
    # Reversal Potential (mV)
    post_XE =   0.0 * np.power(10.0, -3)
    post_IE = -70.0 * np.power(10.0, -3)
    
    # Voltage Thresholds
    post_thReset = -70.0 * np.power(10.0, -3)
    post_thAp    = -30.0 * np.power(10.0, -3)
    post_vAp     =  50.0 * np.power(10.0, -3)  # Peak Action Potential voltage
    
    # Specific Capacitance
    post_cm = 10 * np.power(10.0, 2) 
    
    post_v = np.zeros(shape = (t_stop, 1))
    t_lastSpike = 0
    
    # Membrane Potential
    for time in np.arange(t_stop):
 
        rates = post_instRates[:, time]
        timeSinceLastSpike = time - t_lastSpike 
       
        Xconductance = (rates * pre_preferredStim.T).sum() / pre_rMax
        Iconductance = rates.sum() / pre_rMax 
        
        vInf = (Xconductance * post_XE) + (Iconductance * post_IE) \
               / (Xconductance + Iconductance)
               
        vXexp = Xconductance * (post_thReset - post_XE) * \
                np.exp(-(Xconductance + Iconductance) * timeSinceLastSpike / post_cm)
        
        vIexp = Iconductance * (post_thReset - post_IE) * \
                np.exp(-(Xconductance + Iconductance) * timeSinceLastSpike / post_cm)
               
#        print ("time=%i, timeSincelastSpike=%i, Xcon=%f, Icon=%f, vInf=%f, vXexp=%f, vIexp=%f" \
#               %(time, timeSinceLastSpike, Xconductance, Iconductance, vInf, vXexp, vIexp))
               
        post_v[time] = vInf + vXexp + vIexp
        
        if post_v[time] >= post_thAp:
            post_v[time] = post_vAp
            t_lastSpike = time
            #raw_input()
            
    # Plot post-synaptic Neuron Voltage
    plt.figure("Post Synaptic Membrane Potential")
    plt.title("Post Synaptic Membrane Potential, Stimulus %i, Avg Stimulus Est %f" \
              %(s, np.mean(post_sEstML[post_windowLen/2:])))
    plt.plot(np.arange(t_stop), post_v * np.power(10.0,3))
    plt.plot(np.arange(t_stop), post_thAp * np.ones(shape = (t_stop, 1)) *np.power(10.0,3), 'r', label = 'AP Thresh')
    ax = plt.gca()
    ax.set_xlabel('Time(ms)', x = 1)
    ax.set_ylabel("Membrane Potential(mV)")
    ax.set_ylim([-300, 100])
    ax.legend()
    
    f, (ax1, ax2) = plt.subplots(2, 1)
    plt.title("Single AP profile [stimulus=%s]" %(s))
    tRange = 1000;
    ax1.plot(np.arange(tRange), post_v[0:tRange] * np.power(10.0,3))
    ax2.plot(np.arange(tRange), post_v[0:tRange] * np.power(10.0,3))
    plt.ylim([-300,100])
    ax2.axhline(y = -30 * np.power(10,-3), linewidth=2, color='r')
    
    
    
    #Firing Rate
#    totalCond = Xconductance + Iconductance
#    
#    post_tIsi = post_cm * \
#                (totalCond * ((np.log(totalCond) * post_thReset) - post_thAp)) + \
#                ( Xconductance * post_XE * ( 1 - np.log(totalCond)) ) + \
#                ( Iconductance * post_IE * ( 1 - np.log(totalCond)) ) \
#                / \
#                (totalCond * ( post_thReset*totalCond - Xconductance*post_XE - Iconductance*post_IE) )
#                
#    print post_tIsi           
     
    
     
    
    

if __name__ == "__main__":
    
    #pyplot interactive mode - do not need to close windows after each run
    plt.ion()
    main()
    
   