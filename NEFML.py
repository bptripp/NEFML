"""--------------------------------------------------------------------------------------
Created on Thu May 29 18:24:20 2014

@author: salman Khan

     MAXIMUM LIKLIEHOOD DECODER 1 Dimensional

--------------------------------------------------------------------------------------"""
import numpy as np
import matplotlib.pyplot as plt

def main(s):
    
    ''' -------------------------------------------------------------
    INPUT
    --------------------------------------------------------------'''
    
    ''' Stimulus '''
    s_range = 100
    #TODO add some error checks
    print ("Input Stimulus %i" %(s))
    
    
    ''' Time '''
    t_stop = 1000 # time im ms
    
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
    post_thAp    = -50.0 * np.power(10.0, -3)
    post_vAp     =  50.0 * np.power(10.0, -3)  # Peak Action Potential voltage
    
    # Specific Capacitance
    post_cm = 10 * np.power(10.0, 3) 
    
    post_v = np.zeros(shape = (t_stop, 1))
    post_rateCalc = np.zeros(shape = (t_stop, 1))
    
    # Membrane Potential
    t_lastSpike = 0
    post_nSpikes = 0
    for time in np.arange(t_stop):
 
        rates = post_instRates[:, time]
        timeLastSpike = time - t_lastSpike 
       
        Xcon = (rates * pre_preferredStim.T).sum() 
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
    
    # Plot post-synaptic Neuron Voltage
    plt.figure("Post Synaptic Membrane Potential")
    plt.title("Post Synaptic Membrane Potential, Stimulus %i, Avg Stimulus Est %f" \
              %(s, np.mean(post_sEstML[post_windowLen/2:])))
    plt.plot(np.arange(t_stop), post_v * np.power(10.0,3))
    plt.plot(np.arange(t_stop), post_thAp * np.ones(shape = (t_stop, 1)) * np.power(10.0,3), 'r', label = 'AP Thresh')
    ax = plt.gca()
    ax.set_xlabel('Time(ms)', x = 1)
    ax.set_ylabel("Membrane Potential(mV)")
    ax.legend()
    
    plt.figure("Post Synaptic Firing Rates")
    plt.plot(np.arange(t_stop), post_rateCalc, label='Calculated Rate')
    ax = plt.gca()
    ax.set_xlabel('Time(ms)', x = 1)
    ax.set_ylabel("Firing Rate")
    plt.axhline(y=post_rateMeas, color='red', linewidth=2, label='Measured Rate')
    plt.legend()
    
    return(post_rateMeas, post_rateCalcMean, post_rateCalcVar)

    
       

if __name__ == "__main__":
    
    #pyplot interactive mode - do not need to close windows after each run
    plt.ion()
    stimulus_set = np.arange(100)
    #stimulus_set = [80]
    
    measRate = np.zeros(shape = len(stimulus_set))
    calcRate = np.zeros(shape = len(stimulus_set))
    calcRateVar = np.zeros(shape = len(stimulus_set))
    
    for idx, stimulus in enumerate(stimulus_set):
        measRate[idx], calcRate[idx], calcRateVar[idx] = main(stimulus)
        
    plt.figure("Post Synaptic Neuro")
    plt.title("PostSynaptic Neuron Tuning Curve (Conductance Model)")
    plt.plot(stimulus_set, measRate, label = 'Mean Firing Rate (measured)')
    plt.plot(stimulus_set, calcRate, 'r--', label = 'Mean FiringRate (calculated)' )
    ax = plt.gca()
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Firing Rate')
    plt.legend()
    
    # Plot Variance Vertical Lines
    for idx, stimulus in enumerate(stimulus_set):
        plt.plot([stimulus, stimulus], [(calcRate[idx] + calcRateVar[idx]), \
                                        (calcRate[idx] - calcRateVar[idx])], 'k')
                                        
    plt.figure('Post Synaptic Response Variance')
    ax = plt.gca()
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Variance')
    plt.plot(stimulus_set, calcRateVar)
    plt.title('Post Synaptic Firing Rate Variance')
                                        
    
    
    
    
   