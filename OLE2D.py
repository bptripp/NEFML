'''
  OPTIMUM LINEAR DECODER

- Poisson Firing
- Instantaneous Firing Rates based on sliding average using a linear filter 
- Alpha Function used for sliding window
'''

import numpy as np
import matplotlib.pyplot as plt

def main(s_angle):    
    ''' ---------------------------------------------------------
    INPUT
    -----------------------------------------------------------'''
    s_angle = s_angle * np.pi / 180
    
    ''' Stimulus '''
    # random Stimulus
    #s_angle = np.random.uniform (0, (2 * np.pi))
    
    s = np.array( [np.cos(s_angle), np.sin(s_angle)] )
    
    # Setup Direction Vector Figure and plot stimulus
#    plt.figure("2D space", figsize = (8,8))
#    plt.plot([0, s[0]], [0, s[1]], color = "red", linewidth = 2, \
#    label = 'stimulus')
#    plt.xlim([-1.1, 1.1])
#    plt.ylim([-1.1, 1.1])
#    fig1Ax = plt.gca()
#    fig1Ax.spines['right'].set_color('none')
#    fig1Ax.spines['top'].set_color('none')
#    fig1Ax.xaxis.set_ticks_position('bottom')
#    fig1Ax.spines['bottom'].set_position(('data',0))
#    fig1Ax.yaxis.set_ticks_position('left')
#    fig1Ax.spines['left'].set_position(('data',0))
#    fig1Ax.set_title("Preferred Stimulus Angle %f" \
#    %(np.arctan2(s[1], s[0])*180/np.pi))
    
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
    
    pre_preferredStim = np.zeros(shape = (pre_n, 2))
    pre_preferredStim[:, 0] = np.cos(pre_preferredStimAngle)
    pre_preferredStim[:, 1] = np.sin(pre_preferredStimAngle)
    
#    plt.figure("2D space")    
#    for x, y in pre_preferredStim: 
#        plt.plot([0, x], [0, y], color = "blue", linewidth = 0.5, \
#        linestyle = '--', label = 'preferred stimulus')

    ''' Firing Rates '''
    # Assume cosine tuning curves with firing rate = projection onto preferred 
    # stimulus vector for each neuron
    # Assume zero background firing to get half wave cosine tuning curves
    # For neuron i, raw firing rate, Ri = Bi + Ki(V.Ci)
    pre_fireRates = np.dot(s, pre_preferredStim.T) * pre_rMax
    pre_fireRates[pre_fireRates < 0] = 0
    
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
    
        if rate != 0:
            nSpikes = np.floor( 2 * t_stop * rate / 1000.0)
            spikeTimes = np.cumsum( \
            np.random.exponential(scale = 1000.0/rate, size = nSpikes) )

        pre_spikeTimes.append(spikeTimes)
    
    # Convert to indices and eliminate invalid entries
    for row in np.arange( len(pre_spikeTimes) ):
        pre_spikeTimes[row] = \
        [ int(x) for x in pre_spikeTimes[row] if x < t_stop ] 

    # Print number of spikes for each neuron
#    for idx, item in enumerate(pre_spikeTimes):
#        print "preferred Stimulus:", pre_preferredStimAngle[idx]*180/np.pi, \
#        "Rate: ",len(item)

    pre_out = np.zeros(shape =(pre_n, t_stop))
    
    for row in np.arange(len(pre_spikeTimes)):
        pre_out[row, pre_spikeTimes[row]] = 1 
    
#    fig3 = plt.figure("PostSynaptic Input Spikes")
#    plt.title("PostSynaptic Input Spikes")
#    plt.yticks(np.arange(pre_n))
#    plt.ylabel('Neuron Index')
#    plt.xlabel('Time(ms)')
#    im = plt.imshow(pre_out, aspect = 'auto', cmap = plt.cm.binary)
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
    Optimum Linear Decoding ----------------------------------
    '''
    
    ''' Tuning Curves for all synapses '''
    post_nStim = 360    # Assume 360 input vectors are possible
    post_nSynapses = pre_n  
    
    tmp = np.linspace(0, 2*np.pi, post_nStim, endpoint = False)
    
     # Set of all possible stimuli
    post_stimuli = np.zeros(shape = (post_nStim, 2))
    post_stimuli[:, 0] = np.cos(tmp)
    post_stimuli[:, 1] = np.sin(tmp)
    
    post_tuningCurves = np.zeros(shape = (post_nSynapses, post_nStim))
    
    for idx, stimulus in enumerate(post_stimuli):
        # Dot product
        post_tuningCurves[:, idx] = \
        (stimulus * pre_preferredStim).sum(axis = 1) * pre_rMax
        # Rectification - eliminate -ve rates
        post_tuningCurves[post_tuningCurves < 0] = 0    
    
    # Plot tuning curves of all synapses
#    plt.figure("Tuning Curves")
#    plt.title("Tuning Curves")
#    plt.xlabel("Stimulus")
#    plt.xlim((0,360))
#    plt.ylabel("Firing Rate")
#    for synapse in np.arange(post_nSynapses):
#        plt.plot( np.arange(post_nStim), post_tuningCurves[synapse,:])

    ''' Correlation Matrix '''
    # Normalize Firing Rates & Tuning curves
    post_tuningCurves = post_tuningCurves / pre_rMax
    post_instRates = post_instRates / pre_rMax
     
    post_corrMat = np.zeros(shape = (post_nSynapses, post_nSynapses))
    
    for ii in np.arange(post_nSynapses):
       for jj in np.arange(post_nSynapses):
            post_corrMat[ii, jj] = np.correlate(post_tuningCurves[ii, :], \
            post_tuningCurves[jj, :]) / post_nStim
    # This is not exactly a full corelation, it is just some of each term in 
    # V1 multiplied with corrosponding term in V2. The correlation term if 
    # both vectors were on top of each other, but this is what is needed here
    
    """ TODO: Add variance of Poisson Firing Process """
    post_synapseVar = 100
    post_corrMat = post_corrMat + (post_synapseVar * np.eye(post_nSynapses))

    # Plot the correlation Matrix
#    plt.figure("Firing Rates Correlation Matrix")
#    plt.title("Firing Rates Correlation Matrix")
#    plt.imshow(post_corrMat)
#    plt.colorbar()

    ''' Center of Mass Vectors '''
    # 2 = number of dimensions of the stimulus    
    post_centerOfMass = np.zeros(shape = (post_nSynapses, 2) )
    
    for ii in np.arange(post_nSynapses):
        post_centerOfMass[ii, 0] = \
        (post_tuningCurves[ii, :] * post_stimuli[:, 0]).sum()
        post_centerOfMass[ii, 1] = \
        (post_tuningCurves[ii, :] * post_stimuli[:, 1]).sum()    
    
    # Normalize Center of Mass Vectors
    post_centerOfMass = post_centerOfMass / post_nStim
    
    # Plot Center of Mass Vectors
#    plt.figure("2D space")
#    for x, y in post_centerOfMass:
#        plt.plot([0, x], [0, y], 'g.')
    
    ''' Opimum Linear Decoders '''
    post_invCorrMat = np.linalg.pinv(post_corrMat)
     
    post_oleDecoders = np.zeros(shape = (post_nSynapses, 2))
    for ii in np.arange(post_nSynapses):
        post_oleDecoders[ii, 0] = (post_invCorrMat[ii,:] * post_centerOfMass[:, 0]).sum()
        post_oleDecoders[ii, 1] = (post_invCorrMat[ii,:] * post_centerOfMass[:, 1]).sum()
        
    # Plot post_oleDecoders
#    plt.figure("2D space")
#    for x,y in post_oleDecoders:
#        plt.plot([0,x],[0,y], 'm')        
    
    ''' Stimulus Estimate '''
    tmp = np.zeros(shape = pre_preferredStim.shape)
    post_sEstOle = np.zeros(shape = (t_stop, 2))

    for time in np.arange(t_stop):
        rates = post_instRates[:, time]
        for synapse, rate in enumerate(rates):
             tmp[synapse] = rate * post_oleDecoders[synapse]
        post_sEstOle[time] = tmp.sum(axis = 0)
        
    # Plot Stimulus Estimates
#    plt.figure("2D space")
##    for x,y in post_sEstOle:
##        plt.plot([0,x],[0,y], 'm')  

    ''' Error Estimation '''
    post_oleStimEstDot = np.dot(post_sEstOle, s)
    post_oleEstMag = [np.linalg.norm(estimate) for estimate in post_sEstOle]
    
    post_oldDirectionErr = np.zeros(shape = post_oleStimEstDot.shape)
    for time in np.arange(t_stop):
        post_oldDirectionErr[time] =  (180/ np.pi) * np.arccos( \
        post_oleStimEstDot[time] / (post_oleEstMag[time] * np.linalg.norm(s)) )

    # Plot Estimate Error
#    plt.figure("Error Estimate")
#    plt.plot(np.arange(t_stop), post_oldDirectionErr)
    
    # skip the first few estimates as the firing rate estimate will be incorrect
    post_OleErrorAvg = np.mean(post_oldDirectionErr[post_windowLen:])
    print "Stimulus %f, Average Error %f" %( (s_angle*180/np.pi), post_OleErrorAvg)
    
    return post_OleErrorAvg
 
    
if __name__ == "__main__":
    # if you call this script from the command line (the shell) it will
    # run the 'main' function
        
    #pyplot interactive mode - do not need to close windows after each run
    plt.ion()
    
    s_angle = np.linspace(-180, 180, num = 360, endpoint = False)
    #s_angle = [ 45]
    
    Error = np.zeros(shape = (len(s_angle), 1))
   
    for idx, stimulus in enumerate(s_angle):
        Error[idx] = main(stimulus)
    
    plt.figure("Avg Error")
    plt.plot(s_angle, Error, '+r-')
