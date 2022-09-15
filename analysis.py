# data location
# /data/backed_up/shared/Quantum_data/EEG_data/preproc_EEG
import mne
import numpy as np 
from mne.time_frequency import tfr_morlet
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import entropy #this looks right https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
import pandas as pd
import matplotlib.pyplot as plt
import numpy.matlib as mb

# function to write AFNI style stim timing file
def write_stimtime(filepath, inputvec):
	''' short hand function to write AFNI style stimtime'''
	#if os.path.isfile(filepath) is False:
	f = open(filepath, 'w')
	for val in inputvec[0]:
		if val =='*':
			f.write(val + '\n')
		else:
			# this is to dealt with some weird formating issue
			#f.write(np.array2string(val))
			#print np.array2string(np.around(val,4))
			f.write(np.array2string(np.around(val,6), suppress_small=True).replace('\n','').replace(']','').replace('[','') + '\n') 
			#np.array2string(np.around(val,6), suppress_small=True).replace('\n','').replace(']','').replace('[','') + '\n'
			#np.array2string(np.around(val,6), suppress_small=True).replace('\n','').replace(']','').replace('[','')[2:-1] + '\n'
	f.close()


def extract_model_inputs(df):
  # create new column variable of "run number" 
  cur_df = df
  tState = np.where(cur_df['state']==-1, 1, 0) # recode state (1 now 0 ... -1 now 1 ... this flip matches model better than just changing -1)
  tTask = np.zeros(len(tState)) # face=1, scene=0
  TaskPerformed = np.zeros(len(tState)) # face=1, scene=0

  cue_color = np.array(cur_df['cue']).flatten() # recode cue_color (now -1 (red) is 0 ... and 1 (blue) is 1 )
  cue_color = np.where(cue_color==-1, 0, cue_color) # recode cue_color (now -1 (red) is 0 ... and 1 (blue) is 1 )
  # now change amb to match tProp setup
  tProp = np.array(cur_df['amb']).flatten() # set as proportion of red

  tResp=[]
  target = list(cur_df['target'])
  correct = np.array(cur_df['correct']).flatten()
  subj_resp = np.array(cur_df['subj_resp']).flatten()
  for trl, trl_target in enumerate(target):
      if trl_target == "face":
          tTask[trl] = 1 # change task to 1
      if correct[trl] == 1:
          tResp.append(0)
      else:
          if subj_resp[trl]==-1:
              # no response... cound as wrong task for now
              tResp.append(2)
          elif trl_target == "face":
              # resp should be 0 or 1
              if subj_resp[trl] > 1:
                  # wrong task
                  tResp.append(2)
              else:
                  # right task but wrong answer
                  tResp.append(1)
          elif trl_target == "scene":
              # resp should be 2 or 3
              if subj_resp[trl] < 2:
                  # wrong task
                  tResp.append(2)
              else:
                  # right task but wrong answer
                  tResp.append(1)
      if subj_resp[trl] < 2:
          # they think they should do the scene task
          TaskPerformed[trl]=1
  tResp = np.array(tResp).flatten()

  print("\ntState:\n", tState)
  print("\ntTask:\n", tTask)
  print("\ntProp:\n", tProp)
  print("\ntResp:\n", tResp)

  return tState, tTask, tProp, tResp


def SequenceSimulation(nTrials, switchRange, thetas):

  tState = np.array([])
  cState = 0
  l = 0
  #r = np.array(range(switchRange[0],(switchRange[1]+1),1)) 

  # set up state based on switch range
  while l < nTrials:
    #r = r[np.random.permutation(len(r))]
    #tState = np.concatenate( (tState, mb.repmat(cState, 1, r[0]).flatten()), axis=None )
    r = np.random.choice(switchRange)
    tState = np.concatenate( (tState, mb.repmat(cState, 1, r).flatten()), axis=None )
    cState = 1 - cState
    l = l + r
    #l = l + r[0]

  tState = tState[:nTrials] # make sure tState is only as long as the number of trials
  

  ### new way to create perceptual trial sequence
  tProp = np.array([])
  start_amb = 5
  l = 0
  ambs =[]
  while l < nTrials:
    if l ==0:
      amb_state = start_amb
      ambs.append(amb_state)
    elif l==1:
      amb_state = 4
      ambs.append(amb_state)
    elif l==2:
      amb_state = 3
      ambs.append(amb_state)

    r = np.random.choice([5,12])
    tProp = np.concatenate( (tProp, mb.repmat(amb_state, 1, r).flatten()), axis=None )
    
    if l > 2:
      if amb_state == 5:
        amb_state = 4
      elif amb_state == 1:
        amb_state = 2
      elif ambs[-2] == 5:
        amb_state = 3
      elif ambs[-2] ==1:
        amb_state = 3
      else:
        amb_state = amb_state + np.random.choice([-1,1])
      ambs.append(amb_state)
    
    l = l + r
    #l = l + r[0]
  tProp= tProp[:nTrials]
  #plt.plot(tProp, 'blue', tState, 'black')

  #tProp = np.random.rand(nTrials) # randomly generate proportion (cue)... btw 0 and 1
  possible_prop={5:[0.65, 0.66, 0.67, 0.68,0.35, 0.34, 0.33, 0.32] ,4:[0.61, 0.62, 0.63, 0.64, 0.39, 0.38, 0.37, 0.36], 3:[0.58, 0.59, 0.60, 0.42, 0.41, 0.40], 2:[0.55, 0.56, 0.57, 0.45, 0.44, 0.43],1:[0.51, 0.52, 0.53, 0.49, 0.48, 0.47]}#= [0.11111111,0.33333333,0.44444444,0.48,0.52,0.55555556,0.66666667,0.88888889]
  for i in range(len(tProp)):
    tProp[i] = np.random.choice(possible_prop[tProp[i]])

  # smooth_Prop =moving_average(tProp, 4) #smoothing window of 7?
  # nprop = tProp
  # nprop[0:nTrials-3] = smooth_Prop
  # tProp = nprop   
  plt.plot(abs(0.5-np.array(tProp)), 'blue', tState, 'black')
  
  # avoid extreme values in simulation
  #tProp[tProp < 0.05] = 0.05
  #tProp[tProp > 0.95] = 0.95

  tResp = np.zeros((tState.shape))
  tTask = np.zeros((tState.shape))

  # loop through all trials
  for i in range(nTrials):
    #np.random.shuffle(possible_prop)
    #tProp[i] = possible_prop[0]

    x = np.exp(thetas[0]) * np.log( (tProp[i]/(1 - tProp[i])) )
    p = 1 / (1 + np.exp((-1*x)))

    # agent got confused within 3 trials after switch
    if (i > 4) and ( (tState[i] != tState[(i-1)]) or (tState[i] != tState[(i-2)]) or (tState[i] != tState[(i-3)])):
      # if we are within 3 trials of a switch then assume they're still doing the old task
      cState = 1 - tState[i]
    else:
      # if we are within the first 3 task trials assume they are randomly guessing
      if i < 4:
        if np.random.rand() < 0.5:
          cState = 0
        else:
          cState = 1
      else:
        cState = tState[i]

    if ( (tProp[i] > 0.5) and (tState[i] == 0) ) or ( (tProp[i] < 0.5) and (tState[i] > 0) ):
      tTask[i] = 0
    else:
      tTask[i] = 1

    # rTask is the simulated task the agent thinks it should do
    if np.random.rand() < p:
      #color 0
      if cState == 0:
        rTask = 0
      else:
        rTask = 1
    else:
      #color 1
      if cState == 0:
        rTask = 1
      else:
        rTask = 0

    if tTask[i] != rTask:
      tResp[i] = 2
    else:
      if rTask == 0:
        if np.random.rand() < thetas[1]:
          tResp[i] = 1
        else:
          tResp[i] = 0
      else:
        if np.random.rand() < thetas[2]:
          tResp[i] = 1
        else:
          tResp[i] = 0

  return tState, tProp, tResp, tTask


# this is the model, I modified the JD to export trial by trail posterior distribution
def MPEModel(tState, tProp, tResp):
    
    sE = []
    tE = []

    #start with uniform prior
    #pRange = cell(1, 4);
    pRange = {'lfsp':np.linspace(-5, 5, 201, endpoint=True), 'fter':np.linspace(0, 0.5, 51, endpoint=True), 'ster':np.linspace(0, 0.5, 51, endpoint=True), 'diff_at':np.linspace(0, 0.5, 51, endpoint=True)}
    #range of logistic function slope parameter
    #pRange{1} = -5:0.05:5;
    #range of face task error rate
    #pRange{2} = 0:0.01:0.2;
    #range of scene task error rate
    #pRange{3} = 0:0.01:0.2;
    #diffusion across trials
    #pRange{4} = 0:0.01:0.5;
    #dim = [2, length(pRange{1}), length(pRange{2}), length(pRange{3}), length(pRange{4})];
    dim = np.array( [2, len(pRange['lfsp']), len(pRange['fter']), len(pRange['ster']), len(pRange['diff_at'])] )
    
    total = 1
    # for i = 1:length(dim)
    # total = total * dim(i);
    # end
    # jd = ones(dim) / total;
    for c_dim in dim:
        total = total * c_dim
    jd = np.ones(dim) / total   # the division of total is to normalize values into probability
    
    # to store trial by trial jd
    #all_jd = np.ones(np.array([len(tState), 2, len(pRange['lfsp'])]))

    #likelihood of getting correct response
    #ll = zeros(size(jd));
    ll = np.zeros(jd.shape)

    #tProp = log(tProp ./ (1 - tProp));

    ###############????????????????????????????????????????????????????######
    # Question 3, here why taking the log of ratio of dots for the two colors?
    # Make it more normal?
    ###############????????????????????????????????????????????????????######
    tProp = np.log(tProp / (1 - tProp))  

    # for i = 1 : length(tState)
    # if mod(i, 20) == 0
    #     disp([num2str(i) ' trials have been simulated.']);
    # end
    for it in range(len(tState)):
        if (it % 50) == 0:
            print(str(it), ' trials have been simulated.')
        #diffusion of joint distribution over trials
        # for i4 = 1 : dim(4)
        #     x = (jd(1, :, :, :, i4) + jd(2, :, :, :, i4)) / 2;

        #     jd(1, :, :, :, i4) = jd(1, :, :, :, i4) * (1 - pRange{4}(i4)) + pRange{4}(i4) * x;
        #     jd(2, :, :, :, i4) = jd(2, :, :, :, i4) * (1 - pRange{4}(i4)) + pRange{4}(i4) * x;
        # end

        ###############????????????????????????????????????????????????????######    
        ###############????????????????????????????????????????????????????######
        # Question 4, what is the purpose of this block of code below? 
        # it appears the effect is essentially to smooth paramters between task state 0 and 1??
        
        for i4 in range(dim[4]):
            x = (jd[0, :, :, :, i4] + jd[1, :, :, :, i4]) / 2

            jd[0, :, :, :, i4] = jd[0, :, :, :, i4] * (1 - pRange['diff_at'][i4]) + pRange['diff_at'][i4] * x
            jd[1, :, :, :, i4] = jd[1, :, :, :, i4] * (1 - pRange['diff_at'][i4]) + pRange['diff_at'][i4] * x
        #add estimate as marginalized distribution
        #sE(end + 1) = sum(sum(sum(sum(jd(1, :, :, :, :)))));
        sE.append(np.sum(jd[0, :, :, :, :]))
        ###############????????????????????????????????????????????????????######
        ###############????????????????????????????????????????????????????######


        #first color logit greater than 0 means it is dominant, less than 0
        #means other color dominant

        ############################################
        #### below is the mapping between color ratio, task state, and task
        # tState = 0, tProp<0 : tTask = 1
        # tState = 0, tProp>0 : tTask = 0 
        # tState = 1, tProp<0 : tTask = 0
        # tState = 1, tProp<0 : tTask = 1
        if ((tProp[it] < 0) and (tState[it] > 0)) or ((tProp[it] > 0) and (tState[it] == 0)):
            tTask = 0
        else:
            tTask = 1
        ################################################


        # S_Theta1 = squeeze(sum(sum(sum(jd, 3), 4), 5));
        # tE(end + 1) = 0;
        # here marginalize theta1
        S_Theta1 = np.squeeze(np.sum(np.sum(np.sum(jd, 4), 3), 2)) # double check that axes are correct
        
        tE.append(0)
        for i0 in range(dim[0]):
            for i1 in range(dim[1]):

                ###############????????????????????????????????????????????????????######
                ###############????????????????????????????????????????????????????######
                # Question 5:
                # here this block to logit transform  color ratio to probability of color decision.
                # I believe the formula from JF's white borad was:
                # p(pColor | tProp) = 1 / (1 + exp( -1* theta1 * log(tProp) ))
                #  
                # and convert the color decision probability to task belief (0 to 1)
                # The last line task multiply with S_Theta1 (the marginalized set belief and logistic function slope),
                # is it to weight the task belief by the probability of each logistic slope param in the distribution for each task set...?

                                                     # Quesion 6 here   
                theta1 = np.exp(pRange['lfsp'][i1])  # why take the exponent of logisitc function slope??
                pColor = 1 / (1 + np.exp((-1*theta1) * tProp[it]))
                if i0 == 0:
                    pTask0 = pColor
                else:
                    pTask0 = 1 - pColor

                #print(tE[-1] + pTask0 * S_Theta1[i0, i1])
                tE[-1] = tE[-1] + pTask0 * S_Theta1[i0, i1] 
                ###############????????????????????????????????????????????????????######
                ###############????????????????????????????????????????????????????######


                for i2 in range(dim[2]):
                    for i3 in range(dim[3]):
                        #pP is the likelihood, change this if there is only one type of error
                        # this is for separated response error and task error
                        
                        ###############????????????????????????????????????????????????????######
                        ###############????????????????????????????????????????????????????######
                        # question 7
                        # dont quite get what the liklihood here means given the three typoes of response below
                        # trial-wise response: 0 = correct, 1 = correct task wrong answer, 2 = wrong task
                        if tTask == 0:
                            pP = [pTask0 * (1 - pRange['fter'][i2]), pTask0 * pRange['fter'][i2], (1 - pTask0)]
                        else:
                            pP = [(1 - pTask0) * (1 - pRange['ster'][i3]), (1 - pTask0) * pRange['ster'][i3], pTask0]
                            #print(pP)
                        
                        #posterior, jd now is prior
                        ######
                        # question 8
                        # Here it appears you are using one of the three liklihood to update all joint thetas, will have to ask
                        # JF the logic of othis operation
                        ll[i0, i1, i2, i3, :] = jd[i0, i1, i2, i3, :] * pP[int(tResp[it])]
                        ###############????????????????????????????????????????????????????######
                        ###############????????????????????????????????????????????????????######

                        # #this is for only one type of error
                        # if tTask == 0:
                        #     pP = [pTask0 * (1 - pRange['fter'][i2]), pTask0 * pRange['fter'][i2] + (1 - pTask0)]
                        # else:
                        #     pP = [(1 - pTask0) * (1 - pRange['ster'][i3]), (1 - pTask0) * pRange['ster'][i3] + pTask0]
                        
                        # #posterior, jd now is prior
                        # if tResp(i) == 0:
                        #     ll[i0, i1, i2, i3, :] = jd[i0, i1, i2, i3, :] * pP[0]
                        # else:
                        #     ll[i0, i1, i2, i3, :] = jd[i0, i1, i2, i3, :] * pP[1]
                        
        #normalize
        jd = ll / np.sum(ll)
        #all_jd[it,:,:] = np.sum(jd, axis=(2,3,4))

        mDist={}
        mDist['lfsp'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 4), 3), 2), 0))    # 4 3 2 0
        mDist['fter'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 4), 3), 1), 0))    # 4 3 1 0
        mDist['ster'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 4), 2), 1), 0))    # 4 2 1 0
        mDist['diff_at'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 3), 2), 1), 0)) # 3 2 1 0
    
    return jd, sE, tE, mDist, pRange


#input: thetas are a list of 4 model estimates, same as in model inference
#tState, tProp and tResp are from true trial sequences
#modelBelief is a trial sequence of model belief, in joint distribution of [state, color]
def ModelBeliefGenerateion(thetas, tState, tProp, tResp):
    modelBelief = []
    modelBelief.append(np.ones([2, 2]) / 4)
    
    tProp = np.log(tProp / (1 - tProp))

    jd = modelBelief[-1].copy()
    
    for it in range(len(tState)):
        if (it % 50) == 0:
            print(str(it), ' trials have been simulated.')
        
        for i in range(2):
            x = (jd[0, i] + jd[1, i]) / 2
            jd[0, i] = jd[0, i] * (1 - thetas[3]) + thetas[3] * x
            jd[1, i] = jd[1, i] * (1 - thetas[3]) + thetas[3] * x
            
        
        #first color logit greater than 0 means it is dominant, less than 0
        #means other color dominant
        if ((tProp[it] < 0) and (tState[it] > 0)) or ((tProp[it] > 0) and (tState[it] == 0)):
            tTask = 0
        else:
            tTask = 1

        stateBelief = np.squeeze(np.sum(jd, 1)) 
            
        for i0 in range(2):
            theta1 = np.exp(thetas[0])
            pColor = 1 / (1 + np.exp((-1*theta1) * tProp[it]))
                
            jd[i0, 0] = pColor
            jd[i0, 1] = 1 - pColor
                
            if i0 == 0:
                pTask0 = pColor
            else:
                pTask0 = 1 - pColor

                
            if tTask == 0:
                pP = [pTask0 * (1 - thetas[1]), pTask0 * thetas[1], (1 - pTask0)]
            else:
                pP = [(1 - pTask0) * (1 - thetas[2]), (1 - pTask0) * thetas[2], pTask0]
                        
            #posterior, jd now is prior
            jd[i0, :] = jd[i0, :] * pP[int(tResp[it])]

                        
        jd = jd / np.sum(jd)
        modelBelief.append(jd)
    return modelBelief 

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def MPEModel_Control(tState, tProp, tResp):
    
    sE = []
    tE = []

    #start with uniform prior
    #pRange = cell(1, 4);
    pRange = {'lfsp':np.linspace(-5, 5, 201, endpoint=True), 'fter':np.linspace(0, 0.5, 21, endpoint=True), 'ster':np.linspace(0, 0.5, 21, endpoint=True), 'diff_at':np.linspace(0, 0.5, 21, endpoint=True)}
    #pRange = {'lfsp':np.linspace(-5, 5, 201, endpoint=True), 'fter':np.linspace(0, 0.5, 51, endpoint=True), 
    #          'ster':np.linspace(0, 0.5, 51, endpoint=True), 'diff_at':np.linspace(0, 0.5, 51, endpoint=True)}
    
    #range of logistic function slope parameter
    #pRange{1} = -5:0.05:5;
    #range of face task error rate
    #pRange{2} = 0:0.01:0.2;
    #range of scene task error rate
    #pRange{3} = 0:0.01:0.2;
    #diffusion across trials
    #pRange{4} = 0:0.01:0.5;
    #dim = [2, length(pRange{1}), length(pRange{2}), length(pRange{3}), length(pRange{4})];
    dim = np.array( [2, len(pRange['lfsp']), len(pRange['fter']), len(pRange['ster']), len(pRange['diff_at'])] )

    total = 1
    # for i = 1:length(dim)
    # total = total * dim(i);
    # end
    # jd = ones(dim) / total;
    for c_dim in dim:
        total = total * c_dim
    jd = np.ones(dim) / total
    
    #likelihood of getting correct response
    #ll = zeros(size(jd));
    ll = np.zeros(jd.shape)

    #tProp = log(tProp ./ (1 - tProp));
    tProp = np.log(tProp / (1 - tProp))

    # for i = 1 : length(tState)
    # if mod(i, 20) == 0
    #     disp([num2str(i) ' trials have been simulated.']);
    # end
    for it in range(len(tState)):
        if (it % 50) == 0:
            print(str(it), ' trials have been simulated.')
        #diffusion of joint distribution over trials
        # for i4 = 1 : dim(4)
        #     x = (jd(1, :, :, :, i4) + jd(2, :, :, :, i4)) / 2;

        #     jd(1, :, :, :, i4) = jd(1, :, :, :, i4) * (1 - pRange{4}(i4)) + pRange{4}(i4) * x;
        #     jd(2, :, :, :, i4) = jd(2, :, :, :, i4) * (1 - pRange{4}(i4)) + pRange{4}(i4) * x;
        # end
        for i4 in range(dim[4]):
            x = (jd[0, :, :, :, i4] + jd[1, :, :, :, i4]) / 2

            jd[0, :, :, :, i4] = jd[0, :, :, :, i4] * (1 - pRange['diff_at'][i4]) + pRange['diff_at'][i4] * x
            jd[1, :, :, :, i4] = jd[1, :, :, :, i4] * (1 - pRange['diff_at'][i4]) + pRange['diff_at'][i4] * x
        #add estimate as marginalized distribution
        #sE(end + 1) = sum(sum(sum(sum(jd(1, :, :, :, :)))));
        sE.append(np.sum(jd[0, :, :, :, :]))

        #first color logit greater than 0 means it is dominant, less than 0
        #means other color dominant
        if ((tProp[it] < 0) and (tState[it] > 0)) or ((tProp[it] > 0) and (tState[it] == 0)):
            tTask = 0
        else:
            tTask = 1

        # S_Theta1 = squeeze(sum(sum(sum(jd, 3), 4), 5));
        # tE(end + 1) = 0;
        S_Theta1 = np.squeeze(np.sum(np.sum(np.sum(jd, 4), 3), 2)) # double check that axes are correct
        
        # Making task-set decisions deterministic
        ###############????????????????????????????????????????????????????######
        ###############????????????????????????????????????????????????????######
        # ask JF how this part works to make it determinsitic
        for i0 in range(S_Theta1.shape[1]):
            if S_Theta1[0, i0] > S_Theta1[1, i0]:
                S_Theta1[0, i0] += S_Theta1[1, i0]
                S_Theta1[1, i0] = 0
            else:
                S_Theta1[0, i0] = 0
                S_Theta1[1, i0] += S_Theta1[0, i0]
        ###############????????????????????????????????????????????????????######
        ###############????????????????????????????????????????????????????######
        
        tE.append(0)
        for i0 in range(dim[0]):
            for i1 in range(dim[1]):
                theta1 = np.exp(pRange['lfsp'][i1])
                pColor = 1 / (1 + np.exp((-1*theta1) * tProp[it]))
                
                # Making color perception deterministic
                if pColor > 0.5:
                    pColor = 1
                else:
                    pColor = 0
                
            
                if i0 == 0:
                    pTask0 = pColor
                else:
                    pTask0 = 1 - pColor

                #print(tE[-1] + pTask0 * S_Theta1[i0, i1])
                #print(tE[-1])
                #print(pTask0 * S_Theta1[i0, i1])
                tE[-1] = tE[-1] + pTask0 * S_Theta1[i0, i1] 
                for i2 in range(dim[2]):
                    for i3 in range(dim[3]):
                        #pP is the likelihood, change this if there is only one type of error
                        # this is for separated response error and task error
                        if tTask == 0:
                            pP = [pTask0 * (1 - pRange['fter'][i2]), pTask0 * pRange['fter'][i2], (1 - pTask0)]
                        else:
                            pP = [(1 - pTask0) * (1 - pRange['ster'][i3]), (1 - pTask0) * pRange['ster'][i3], pTask0]
                        
                        #posterior, jd now is prior
                        ll[i0, i1, i2, i3, :] = jd[i0, i1, i2, i3, :] * pP[int(tResp[it])]

                        # #this is for only one type of error
                        # if tTask == 0:
                        #     pP = [pTask0 * (1 - pRange['fter'][i2]), pTask0 * pRange['fter'][i2] + (1 - pTask0)]
                        # else:
                        #     pP = [(1 - pTask0) * (1 - pRange['ster'][i3]), (1 - pTask0) * pRange['ster'][i3] + pTask0]
                        
                        # #posterior, jd now is prior
                        # if tResp(i) == 0:
                        #     ll[i0, i1, i2, i3, :] = jd[i0, i1, i2, i3, :] * pP[0]
                        # else:
                        #     ll[i0, i1, i2, i3, :] = jd[i0, i1, i2, i3, :] * pP[1]
                        
        #normalize
        jd = ll / np.sum(ll)
    
        mDist={}
        mDist['lfsp'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 4), 3), 2), 0))    # 4 3 2 0
        mDist['fter'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 4), 3), 1), 0))    # 4 3 1 0
        mDist['ster'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 4), 2), 1), 0))    # 4 2 1 0
        mDist['diff_at'] = np.squeeze(np.sum(np.sum(np.sum(np.sum(jd, 3), 2), 1), 0)) # 3 2 1 0
    
    return jd, sE, tE, mDist, pRange
  
def mirror_evoke(ep):
	
	e = ep.copy()
	nd = np.concatenate((np.flip(e._data[:,:,e.time_as_index(-1)[0]:e.time_as_index(0)[0]], axis=2), e._data, np.flip(e._data[:,:,e.time_as_index(e.tmax-1)[0]:e.time_as_index(e.tmax)[0]],axis=2)),axis=2)
	tnmin = e.tmin - 1
	tnmax = e.tmax + 1 
	e._set_times(np.arange(tnmin,tnmax+e.times[2]-e.times[1],e.times[2]-e.times[1]))
	e._data = nd

	return e

### model
df = pd.read_csv("/mnt/cifs/rdss/rdss_kahwang/Quantum_Task/Data_EEG/sub-10162/beh/sub-10162_task-Quantum_v2-2_output.csv")
tState, tTask, tProp, tResp = extract_model_inputs(df)
all_jd, sE, tE, mDist, pRange = MPEModel(tState, tProp, tResp)
j_thetas = np.sum(all_jd, axis=0)
pRange = {'lfsp':np.linspace(-5, 5, 201, endpoint=True), 'fter':np.linspace(0, 0.5, 51, endpoint=True), 'ster':np.linspace(0, 0.5, 51, endpoint=True), 'diff_at':np.linspace(0, 0.5, 51, endpoint=True)}
j_theta = [pRange['lfsp'][np.argmax(np.sum(j_thetas,axis=(1,2,3,)))], pRange['fter'][np.argmax(np.sum(j_thetas,axis=(0,2,3)))], pRange['ster'][np.argmax(np.sum(j_thetas,axis=(0,1,3,)))], pRange['diff_at'][np.argmax(np.sum(j_thetas,axis=(0,1,2,)))]]


epch = mne.read_epochs("/data/backed_up/shared/Quantum_data/EEG_data/preproc_EEG/sub-10162_task-Quantum_trl_eeg-epo.fif")
tState = epch.metadata['tState'].values
tProp = epch.metadata['tProp'].values
tResp = epch.metadata['tResp'].values

beliefs = ModelBeliefGenerateion(j_theta, tState, tProp, tResp)

t_entropy = [] #entorpy of dist
t_entropy_change = [0.0] #signed change of entropy from previous trial
for t in range(len(tState)):
  t_entropy.append(entropy(beliefs[t].flatten()))

  if t>0:
    t_entropy_change.append(entropy(beliefs[t].flatten())-entropy(beliefs[t-1].flatten()))

t_entropy = np.array(t_entropy)
t_entropy_change = np.array(t_entropy_change)

td_SE =[0.0]
for t in range(len(tState)):
  if t >0:
    td_SE.append(abs(sE[t]-0.5) - abs(sE[t-1]-0.5))
td_SE = np.array(td_SE)


#tfr
epch = mne.read_epochs("/data/backed_up/shared/Quantum_data/EEG_data/preproc_EEG/sub-10162_task-Quantum_trl_eeg-epo.fif")
epch.info['bads']=[]
freqs = np.logspace(*np.log10([1, 40]), num=20)
n_cycles = 6 #freqs/2  # think about mirror
d = mirror_evoke(mirror_evoke(mirror_evoke(mirror_evoke(mirror_evoke(mirror_evoke(epch))))))
tfr = tfr_morlet(d, freqs=freqs, picks=np.arange(0,65), average=False,n_cycles=n_cycles, use_fft=True, return_itc=False, decim=1, n_jobs=24)
tfr.crop(tmin=-1,tmax=2)
tfr.apply_baseline((-1,0), "zscore")
#tfr.average().plot_topo(vmin=-5, vmax=5)

X = abs(tfr.metadata['MPE_sE'].values-0.5) #t_entropy_change#abs(tfr.metadata['MPE_sT'].values-0.5) #e#abs(tfr.metadata['MPE_sE'].values-0.5)
X = sm.add_constant(X)
data = np.zeros((tfr.data.shape[1],tfr.data.shape[2],tfr.data.shape[3]))
for c in np.arange(tfr.data.shape[1]):
    for f in np.arange(tfr.data.shape[2]):
        for t in np.arange(tfr.data.shape[3]):
            Y = tfr.data[:,c,f,t]
            model = sm.OLS(Y, X).fit()
            data[c,f,t] = model.tvalues[1]

reg_tfr = mne.time_frequency.AverageTFR(data = data, info = tfr.info, times = tfr.times, freqs= tfr.freqs, nave = tfr.data.shape[0])
reg_tfr.plot_joint()


#10162
#reg_tfr.plot(picks=['P4', 'FPz', 'FP1'], tmin=-.2, tmax=2, vmin=-2.5, vmax=2.5, combine='mean')
#reg_tfr.plot_topomap(tmin=0.5,tmax=1, fmin=5,fmax=10)

reg_tfr.plot(picks=['Fz', 'FC2', 'FCz', 'C2', 'C4'], tmin=-.2, tmax=2, vmin=-2.5, vmax=2.5, fmin=2, combine='mean')
reg_tfr.plot_topomap(tmin=0.5,tmax=1, fmin=1,fmax=4)
