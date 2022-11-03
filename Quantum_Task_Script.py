#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Authors: Stephanie C. Leach, Kai Hwang.
Hwang Lab, Dept. of Psychological and Brain Sciences
at the University of Iowa, Iowa City, IA
As of June 15, 2022:
    Office:355N PBSB
    Office Phone:319-467-0610
    Fax Number:319-335-0191
    Lab:355W PBSB
Lab Contact Info: 
    Web - https://kaihwang.github.io/
    Email - kai-hwang@uiowa.edu
"""
# # # ----------- Import functions/packages ------------- # # #
from psychopy import gui, visual, core, data, event, monitors
from psychopy.hardware import keyboard
import numpy as np  # whole numpy lib is available, prepend 'np.'
from random import choice as randomchoice
import random
import os  # handy system and path functions
import sys  # to get file system encoding
import math
import glob # this pulls files in directories to create Dictionaries/Str
import pandas as pd #Facilitates data structure and analysis tools. prepend 'np.'
from datetime import datetime
from PIL import Image
import serial
import csv

######################################################################################################################## 
##################################################   Initialization  ##################################################
# # # ----------- Change Directory ------------- # # #
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))#.decode(sys.getfilesystemencoding())
os.chdir(_thisDir)
output_dir = os.getcwd() #"/Users/scleach/OneDrive - University of Iowa/Research/Research_Projects/human task/"
print(output_dir)

######################################################################################################################
##################################################   Get Subj Info  ##################################################
######## GUI and info entry
# set up for GUI
initial_dict = {'method': ['Behavioral','EEG','MRI']}
dlg0 = gui.DlgFromDict(dictionary=initial_dict, title="Select Method", sortKeys=False)

#subInfo = {'Subject_ID': 99999, 'Block': 1, 'Gender': ['Prefer not to say', 'Male', 'Female', 'Non-Binary', 'Other'], 
#            'Age': 0, 'current_task': ["ColorPerception", "Quantum"], 'method': initial_dict['method']} 
# use gui to get info before starting the experiment
#dlg = gui.DlgFromDict(dictionary=subInfo, title="Subject Info Box", sortKeys=False)
#if dlg.OK == False:
#    core.quit()  # user pressed cancel
#cur_subid = subInfo['Subject_ID'] # pull out subid

if initial_dict['method'] != 'MRI':
    subInfo = {'Subject_ID': 99999, 'Gender': ['Prefer not to say', 'Male', 'Female', 'Non-Binary', 'Other'], 
                'Age': 0, 'current_task': ["Quantum","ColorPerception"], 'method': initial_dict['method']} 
    # use gui to get info before starting the experiment
    dlg = gui.DlgFromDict(dictionary=subInfo, title="Subject Info Box", sortKeys=False)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    cur_subid = subInfo['Subject_ID'] # pull out subid
    
    # load subject info sheet to get and save gui info
    subj_info_df = pd.read_csv("Quantum_Task_Subject_Info.csv")
    ind = len(subj_info_df["Subject_ID"]) # set row as length of rows plus one (avoid overwriting data)
    # SAVE OUT ENTERED INFO BEFORE STARTING TASK
    subj_info_df.loc[ind, 'Subject_ID'] = subInfo['Subject_ID']
    subj_info_df.loc[ind, 'SessionDate'] = datetime.today().strftime('%m/%d/%y') # add a simple timestamp
    subj_info_df.loc[ind, 'StartTime'] = datetime.today().strftime('%I:%M:%S %p')
    subj_info_df.loc[ind, 'Version'] = 3.0
    subj_info_df.loc[ind, 'Method'] = subInfo['method']
    subj_info_df.loc[ind, 'Gender'] = subInfo['Gender']
    subj_info_df.loc[ind, 'Age'] = subInfo['Age']
    #print(subj_info_df)
    subj_info_df.to_csv("Quantum_Task_Subject_Info.csv", index=False)
else:
    subInfo = {'Subject_ID': 99999, 'Block': 1, 'Session': 1, 'current_task': "Quantum", 'method': initial_dict['method']} 
    # use gui to get info before starting the experiment
    dlg = gui.DlgFromDict(dictionary=subInfo, title="Subject Info Box", sortKeys=False)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    cur_subid = subInfo['Subject_ID'] # pull out subid
    
    # load subject info sheet to get and save gui info
    subj_info_df = pd.read_csv("Quantum_Task_Subject_Info.csv")
    ind = len(subj_info_df["Subject_ID"]) # set row as length of rows plus one (avoid overwriting data)
    # SAVE OUT ENTERED INFO BEFORE STARTING TASK
    subj_info_df.loc[ind, 'Subject_ID'] = subInfo['Subject_ID']
    subj_info_df.loc[ind, 'SessionDate'] = datetime.today().strftime('%m/%d/%y') # add a simple timestamp
    subj_info_df.loc[ind, 'StartTime'] = datetime.today().strftime('%I:%M:%S %p')
    subj_info_df.loc[ind, 'Version'] = 3.0
    subj_info_df.loc[ind, 'Method'] = subInfo['method']
    #print(subj_info_df)
    subj_info_df.to_csv("Quantum_Task_Subject_Info.csv", index=False)


max_time = float('inf') # float(inf) would set as inf (or until button press), any other value is okay for testing code
core.wait(2) # pause 2 seconds so we can read output to terminal


testing_for_MRI = 1 # if 1 means to use ITIs for MRI for behavioral



#######################################################################################################################
##################################################   Set up Monitor  ##################################################
##### Define monitor window
if subInfo['method'] == 'Behavioral':
    expInfo = {'win':[1920,1080], 'frameRate':None, 'colorSpace':'rgb', 'backgroundColor':[0,0,0]} # set up to save exp info (task parameters, trial info, etc)
    visual_unit = 'deg' #'pix' #
    mon = monitors.Monitor('Behavioral/EEGRoomMonitor', distance = 70, width = 53.5)
    mon.setSizePix(expInfo['win'])
    win = visual.Window(expInfo['win'], units=visual_unit, fullscr=True, monitor=mon, checkTiming=True, colorSpace = 'rgb', color=[0,0,0])

elif subInfo['method'] == 'EEG':
    expInfo = {'win':[1920,1080], 'frameRate':None, 'colorSpace':'rgb', 'backgroundColor':[0,0,0]} # set up to save exp info (task parameters, trial info, etc)
    visual_unit = 'deg' #'pix' #
    mon = monitors.Monitor('Behavioral/EEGRoomMonitor', distance = 70, width = 53.5)
    mon.setSizePix(expInfo['win'])
    win = visual.Window(expInfo['win'], units=visual_unit, fullscr=True, monitor=mon, checkTiming=True, colorSpace = 'rgb', color=[0,0,0])

else:
    expInfo = {'win':[1440,900], 'frameRate':None, 'colorSpace':'rgb', 'backgroundColor':[0,0,0]} # set up to save exp info (task parameters, trial info, etc)
    visual_unit = 'deg' #'pix' #
    win = visual.Window(expInfo['win'], units=visual_unit, fullscr=True, screen=0, allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb', blendMode='avg', useFBO=True)
win.mouseVisible = False

##### Temporal parameters.
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frame_rate_def = round(expInfo['frameRate'])  # frame_rate_def = 1.0 / round(expInfo['frameRate'])
else:
    frame_rate_def = 60.0  # could not measure, so default 60 Hz   #frame_rate_def = 1.0 / 60.0 
print("frame_rate_def",frame_rate_def)
# Add multiple shutdown keys "at once".
for key in ['q', 'escape']:
    event.globalKeys.add(key, func=core.quit)


######################################################################################################################
########################################  Set up Task Paramters & Keyboard  ##########################################
##### Response keys and keyboard
if subInfo['method'] != 'MRI':
    resp_keys = ['a','s','l','k']
    kb = keyboard.Keyboard() # keyboard has better timing that other keypress functions8
else:
    # ADD MRI RESP CODE
    #resp_keys = ['a','s','l','k']
    #resp_keys = ['2','1','3','4'] # MIGHT CHANGE # for fdad box
    resp_keys = ['1','2','4','3'] # For an UPRIGHT fdad box
    
    #resp_keys = ['8','7','3','2'] # MIGHT CHANGE 
    #resp_keys = ['7','6','2','1'] # MIGHT CHANGE # what Juniper used
    kb = keyboard.Keyboard() # keyboard has better timing that other keypress functions

##### Task AND Trial Parameters
# ------- Quantum ----------------|---- color Perception -----
# Cue ......... 1000 ms           |  Cue ......... 1000 ms   
# Delay ....... 500 ms            |  RespWin ..... 2500 ms  
# Stim ........ 1500 ms           |  ITI ......... 1000-1500 ms   
# Feedback .... 500 ms            |                         
# ITI ......... 2000-3000 ms      |                         
# -------------------------------------------------------------
if subInfo['method'] == 'Behavioral':
    CP_Task_Parameters = {'respKeys': ['r','y'], 'n_blocks': 5, 'n_trials': 75, 
                        'easy':[25,20,15,10,5], 'med':[25,20,15,10,5], 'hard':[15,15,15,15,15], 'harder':[5,10,15,20,25], 'hardest':[5,10,15,20,25]}
    CP_Trl_Durs = {'cue': 1.0, 'resp': 2.5, 'ITI': [2.0, 1.5]}

    Quantum_Task_Parameters = {'respKeys': resp_keys, 'n_blocks': 6, 'n_trials': 40, 'n_prac_trials': 40, 'switchRange': [10,20], 
                                'easy':[10, 5, 0, 0, 0, 0], 
                                'med':[10,10,10, 9, 9, 9], 
                                'hard':[10,10,10,10,10,10], 
                                'harder':[10,10,10,10,10,10], 
                                'hardest':[0, 5, 10,11,11,11]}
    Quantum_Trl_Durs = {'cue': 1.0, 'delay': 0.5, 'stim': 1.5, 'feedback': 0.5, 'ITI': [2.0, 2.1, 2.15, 2.2, 2.3, 2.35, 2.4, 2.5, 2.6, 2.65, 2.7, 2.8, 2.85, 2.9, 3.0]}

elif subInfo['method'] == 'EEG':
    CP_Task_Parameters = {'respKeys': ['r','y'], 'n_blocks': 5, 'n_trials': 75, 
                        'easy':[25,20,15,10,5], 'med':[25,20,15,10,5], 'hard':[15,15,15,15,15], 'harder':[5,10,15,20,25], 'hardest':[5,10,15,20,25]}
    CP_Trl_Durs = {'cue': 1.0, 'resp': 2.5, 'ITI': [2.0, 2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9, 3.0]}

    Quantum_Task_Parameters = {'respKeys': resp_keys, 'n_blocks': 9, 'n_trials': 75, 'n_prac_trials': 45, 'switchRange': [10,30], 
                                'easy':[20,15,15,15,10,5,5,5,5], 
                                'med':[20,20,15,15,15,15,15,10,10], 
                                'hard':[20,20,20,20,20,20,15,15,15], 
                                'harder':[15,15,15,15,15,15,15,15,15], 
                                'hardest':[0,5,10,10,15,20,25,30,30]}
    Quantum_Trl_Durs = {'cue': 1.0, 'delay': 0.5, 'stim': 1.5, 'feedback': 0.5, 'ITI': [2.0, 2.1, 2.15, 2.2, 2.3, 2.35, 2.4, 2.5, 2.6, 2.65, 2.7, 2.8, 2.85, 2.9, 3.0]}
else:
    Quantum_Task_Parameters = {'respKeys': resp_keys, 'n_blocks': 7, 'n_trials': 40, 'n_prac_trials': 40, 'switchRange': [10,20], 
                                'easy':  [10, 5, 5, 3, 2, 0, 0], 
                                'med':   [10,10, 6, 6, 6, 5, 5], 
                                'hard':  [10,10,10,10,10,12,12], 
                                'harder':[10,10,10,10,11,12,12], 
                                'hardest':[0, 5, 9,11,11,11,11]}
    Quantum_Trl_Durs = {'cue': 1.0, 'delay': 0.5, 'stim': 1.5, 'feedback': 0.5, 'ITI': [2.0, 2.1, 2.15, 2.2, 2.3, 2.35, 2.4, 2.5, 2.6, 2.65, 2.7, 2.8, 2.85, 2.9, 3.0]}

ITIs_for_runs = []
try:
    ITIpath = '/Volumes/rdss_kahwang/Generate_ITIs/QuantITIs/' #'Z:/Generate_ITIs/ThalHiITIs/'
    rand_order_list = np.random.permutation(os.listdir(ITIpath))
except:
    ITIpath = 'Z:/Generate_ITIs/QuantITIs/' #'Z:/Generate_ITIs/ThalHiITIs/'
    rand_order_list = np.random.permutation(os.listdir(ITIpath))
# save out itis used
with open(("sub-"+str(subInfo['Subject_ID'])+"_ITI_Files.txt"), "w") as out_txt:
    out_txt.write(str(rand_order_list[:7]))
for cur_run in range(int(Quantum_Task_Parameters['n_blocks'])):
    try:
        ITIpath = '/Volumes/rdss_kahwang/Generate_ITIs/QuantITIs/' #'Z:/Generate_ITIs/ThalHiITIs/'
        ITI_rand_file = rand_order_list[cur_run]
        print(ITI_rand_file)
        ITI_rand_file = open(ITIpath+ITI_rand_file,'r').readlines()
    except:
        ITIpath = 'Z:/Generate_ITIs/QuantITIs/' #'Z:/Generate_ITIs/ThalHiITIs/'
        ITI_rand_file = rand_order_list[cur_run]
        print(ITI_rand_file)
        ITI_rand_file = open(ITIpath+ITI_rand_file,'r').readlines()
    ITI_list = []
    for i in ITI_rand_file:
        t=i.split('\n')
        t=t[0]
        if t == '':
            t=6 # empty spaces at the end of the file to insert pre-defined value
        ITI_list.append(float(t))
    ITIs_for_runs.append(ITI_list)


##### Parameters of the cue array specifically
cue_Params = {'dotSize':6, 'cue_field_size':15, 'n_dots':1000}
frac_params = {-1:{'easy': [[0.65, 0.66, 0.67, 0.68], [0.35, 0.34, 0.33, 0.32]], 
                    'med': [[0.61, 0.62, 0.63, 0.64], [0.39, 0.38, 0.37, 0.36]], 
                    'hard': [[0.58, 0.59, 0.60], [0.42, 0.41, 0.40]], 
                    'harder': [[0.55, 0.56, 0.57], [0.45, 0.44, 0.43]],
                    'hardest': [[0.51, 0.52, 0.53], [0.49, 0.48, 0.47]]},
                1:{'easy': [[0.35, 0.34, 0.33, 0.32], [0.65, 0.66, 0.67, 0.68]],  
                    'med': [[0.39, 0.38, 0.37, 0.36], [0.61, 0.62, 0.63, 0.64]], 
                    'hard': [[0.42, 0.41, 0.40], [0.58, 0.59, 0.60]], 
                    'harder': [[0.45, 0.44, 0.43], [0.55, 0.56, 0.57]],
                    'hardest': [[0.49, 0.48, 0.47], [0.51, 0.52, 0.53]]}}

#### Set up for EEG
if subInfo['method'] == 'EEG':
    trigDict = {'startSaveflag':bytes([201]), 'stopSaveflag':bytes([255]), 'blockEnd':203, 
                'state':{1:{ 1:{'easy':151, 'med':153, 'hard':155, 'harder':157, 'hardest':159}, -1:{'easy':161, 'med':163, 'hard':165, 'harder':167, 'hardest':169} }, 
                        -1:{ 1:{'easy':171, 'med':173, 'hard':175, 'harder':177, 'hardest':179}, -1:{'easy':181, 'med':183, 'hard':185, 'harder':187, 'hardest':189} } }, # grab diff codes depending on state 1 or -1 and cue 1(B) or -1(R)
                'delay':199,  
                'target':{'face':{'female':{'landscape':119, 'city':117}, 'male':{'landscape':139, 'city':137}}, 
                        'scene':{'female':{'landscape':219, 'city':217}, 'male':{'landscape':239, 'city':237}}}, 
                'resp':{0:121, 1:123, 2:129, 3:127, -1:109, 'r':141 , 'y':143},  
                'feedback_ITI':{0:103 , 1:107}}
    port=serial.Serial('COM6',baudrate=115200)
    port.close()
# notes on EEG triggers
# cue trigger will be 3 digits: [color(2digits) proportion_conds] ... [RR|YY][easy|med|hard|hardest]
#    state(1): yellow(1) = 15 ... red(-1) = 16  |   state(-1): yellow(1) = 17 ... red(-1) = 18
#    easy = 1 ... med = 3 ... hard = 5 ... harder = 7 ... hardest = 9
#    EXAMPLES: 151 means state1 yellow cue ... easy proportion
#              185 means state2 red  cue ... hard proportion
# target will be 3 digits: [face/scene female/male landscape/city]
#    face = 1 ... scene = 2
#    female = 1 ... male = 3  ...  city = 7 ... landscape = 9
#    EXAMPLES: 117 means target==face  with female and landscape
#              139 means target==face  with male   and city
#              239 means target==scene with male   and city
#              217 menas target==scene with female and landscape
# response will be 3 digits:
#    resp female = 121  ...  resp male = 123  ...  resp city = 127  ...  resp landscape = 129
#    resp red = 141  ...  resp yellow = 143
# feedback will be 3 digits:
#    plus = 103  ...  minus = 107   ...   no resp = 109

#### Set up text objects and basic shapes like fixation and feedback signs
text = visual.TextStim(win=win, wrapWidth=35)
fixation = visual.GratingStim(win=win, units=visual_unit, size=0.25, pos=[0,0], sf=0, color='black', colorSpace='rgb')
fixation_gray = visual.GratingStim(win=win, units=visual_unit, size=0.3, pos=[0,0], sf=0, color='black', opacity=0.5, colorSpace='rgb')
horz_line = visual.Rect(win=win, units=visual_unit, pos=(0,0), size=[1.5, 0.35], color='black', fillColor='black', lineColor='black', colorSpace='rgb')
vert_line = visual.Rect(win=win, units=visual_unit, pos=(0,0), size=[0.35, 1.5], color='black', fillColor='black', lineColor='black', colorSpace='rgb')

Quantum_Instructions = visual.TextStim(win=win, name='Instruct', text=u'Here is the figure from the instructions sheet\nPress any key to see a slowed down example of just the cue, delay, and stimuli screens', 
                                    font=u'Arial', units='norm', pos=(0,0.82), height=0.08, ori=0, color=u'white', colorSpace='rgb', opacity=1, wrapWidth = 800 )

EndofBlock_Text = visual.TextStim(win=win, text=u'End of trial block: please take a short break or press any key to continue.', font=u'Arial', units='norm', pos=(0, 0), height=0.08, ori=0, color=u'white', colorSpace='rgb', opacity=1)

CheckChoice_Text = visual.TextStim(win=win, name='Choice', text=u'Press [P] to start the practice or\npress [Y] to start the task.\n\nPress [N] to go back through the instructions.', 
                                    font=u'Arial', units='norm',  pos=(0, 0), height=0.08, ori=0, color=u'white', colorSpace='rgb', opacity=1)

if subInfo['method']=='MRI':
    fixation = visual.GratingStim(win=win, units=visual_unit, size=0.3, pos=[0,0], sf=0, color='black', colorSpace='rgb')
    fixation_gray = visual.GratingStim(win=win, units=visual_unit, size=0.3, pos=[0,0], sf=0, color='black', opacity=0.5, colorSpace='rgb')
    horz_line = visual.Rect(win=win, units=visual_unit, pos=(0,0), size=[2.0, 0.5], color='black', fillColor='black', lineColor='black', colorSpace='rgb')
    vert_line = visual.Rect(win=win, units=visual_unit, pos=(0,0), size=[0.5, 2.0], color='black', fillColor='black', lineColor='black', colorSpace='rgb')

    Welc = visual.TextStim(win=win, name='Welc', text=u'Welcome!', units='norm', font=u'Arial', pos=(0, 0), height=0.09, ori=0, color=u'white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0)

    Directions = visual.TextStim(win=win, text=u'You are now about to begin the task. \n\nGet Ready \n\nPress Any Key to Continue',
        font=u'Arial', units='norm', pos=(0, 0), height=0.09, ori=0, color=u'white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0)

    Fix_Cue = visual.TextStim(win=win, text=u'+', units='norm', font=u'Arial', pos=(0, 0), height=0.3, ori=0, color=u'white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0)

    Wait_for_Scanner = visual.TextStim(win=win, text=u'Waiting for MRI to initiate task', units='norm', pos=(0,0), height=0.09, ori=0, color=u'white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0)

################################################################################################################################
##################################################   Define Task functions   ##################################################
def make_ITI(ITI_list, trial_n):
        #ITI=np.random.choice([1,2,3,4,5,6,7,8,9,10],1,p=[(.7/9),(.7/9),(.7/9),.3,(.7/9),(.7/9),(.7/9),(.7/9),(.7/9),(.7/9)])[0] # averages to around 4 seconds?
        ITI=ITI_list[trial_n]
        return ITI

def draw_shapes(list_of_shapes):
    for cur_shape in list_of_shapes:
        cur_shape.draw()
        win.flip()
    #nothing to return

def check_prac_task_choice(textobj,repeat_instructions, do_task):
    # check if they want to repeat instructions or move onto task
    textobj.draw()
    win.flip()
    keys = event.waitKeys(keyList=['p','y','n','escape'])
    for key in keys:
        if key=="y":
            repeat_instructions = False
            do_task = 1
        elif key == "p":
            repeat_instructions = False
            do_task = 0
        elif key == 'escape':
                win.close()
                core.quit()
                
    return repeat_instructions, do_task

def generate_prac_dict(Quantum_Output, Quantum_Task_Parameters, r_add):
    prac_dict = {'trial': np.linspace(1,Quantum_Task_Parameters['n_prac_trials'],num=Quantum_Task_Parameters['n_prac_trials'],endpoint=True), 
                'state': Quantum_Output['state'][r_add:(r_add+Quantum_Task_Parameters['n_prac_trials'])], 
                'cue': Quantum_Output['cue'][r_add:(r_add+Quantum_Task_Parameters['n_prac_trials'])], 
                'cue_predominant_color': Quantum_Output['cue_predominant_color'][r_add:(r_add+Quantum_Task_Parameters['n_prac_trials'])], 
                'amb': [],  'amb_r': np.zeros(Quantum_Task_Parameters['n_prac_trials']), 'amb_y': np.zeros(Quantum_Task_Parameters['n_prac_trials']),
                'target': Quantum_Output['target'][r_add:(r_add+Quantum_Task_Parameters['n_prac_trials'])], 
                'corr_resp': Quantum_Output['corr_resp'][r_add:(r_add+Quantum_Task_Parameters['n_prac_trials'])], 
                'subj_resp': np.ones(Quantum_Task_Parameters['n_prac_trials'])*-1 , 
                'correct': np.zeros(Quantum_Task_Parameters['n_prac_trials']), 
                'RT': np.ones(Quantum_Task_Parameters['n_prac_trials'])*-1}
    return prac_dict

def give_example(list_of_texts_for_screen, list_of_visuals, dict_of_time_durs, frac_params, cue_Params, fixation):
    text = visual.TextStim(win=win, wrapWidth=35)
    for ind, cur_txt in enumerate(list_of_texts_for_screen):
        text.text=(cur_txt)
        if list_of_visuals[ind] == "cue":
            # present cue
            text.pos = (0,8)
            trl_cue = np.random.choice([-1,1])
            trl_amb = 'easy'
            n_red_task = round(frac_params[trl_cue][trl_amb][0][1]*cue_Params['n_dots']) # red first, blue second
            n_blue_task = round(frac_params[trl_cue][trl_amb][1][1]*cue_Params['n_dots']) # red first, blue second
            ### Generate red and blue dots. Color coherence starts from chance and shift to final fraction over time.
            dots_dotstim_red = visual.DotStim(win, fieldShape='circle', fieldSize=cue_Params['cue_field_size'], speed=10, dotLife=100,
                                                nDots=int(n_red_task), coherence=0., signalDots='different', noiseDots='walk',
                                                color='red', dotSize=int(cue_Params['dotSize']))
            dots_dotstim_blue = visual.DotStim(win, fieldShape='circle', fieldSize=cue_Params['cue_field_size'], speed=10, dotLife=100,
                                                nDots=int(n_blue_task), coherence=0., signalDots='different', noiseDots='walk',
                                                color='yellow', dotSize=int(cue_Params['dotSize']))
            for frame in range(int(60*dict_of_time_durs['cue'])):
                text.draw()
                dots_dotstim_red.draw()
                dots_dotstim_blue.draw()
                fixation.draw()
                win.flip()
        elif list_of_visuals[ind] == "delay":
            # just present text
            text.pos = (0,5)
            text.draw()
            win.flip()
            event.waitKeys(maxWait=dict_of_time_durs['delay']) # wait for them to decide they know the answer
        elif list_of_visuals[ind] == "task":
            # present face and building stimuli
            text.pos = (0,8)
            text.draw()
            (Img_Dict[np.random.randint(0,5)]['Scene_img']).draw()
            (Img_Dict[np.random.randint(0,5)]['Face_img']).draw()
            win.flip()
            event.waitKeys(maxWait=dict_of_time_durs['task'])
        elif list_of_visuals[ind] == "resp":
            text.pos = (0,8)
            text.draw()
            win.flip()
            event.waitKeys(maxWait=dict_of_time_durs['resp'])
        elif list_of_visuals[ind] == "feedback":
            text.pos = (0,8)
            text.draw()
            horz_line.draw()
            win.flip()
            event.waitKeys(maxWait=dict_of_time_durs['feedback'])
        else:
            text.pos = (0,0)
            text.draw()
            win.flip()
            event.waitKeys()
    #nothing to return

def disp_block_text(EndofBlock_Text, cur_block_acc, lastblock, max_time):
    if lastblock:
         EndofBlock_Text.text = "Task complete: please press any key to end."
    else: 
        if cur_block_acc > 0:
            EndofBlock_Text.text = "End of trial block: please take a short break or press any key to continue.\n\n\nAccuracy for last block was " + str(cur_block_acc) + "%"
        else:
            EndofBlock_Text.text = "End of trial block: please take a short break or press any key to continue."
    EndofBlock_Text.draw()
    win.flip()
    core.wait(1)
    event.waitKeys(maxWait=max_time)
    #nothing to return

def give_intertrial_text(text_obj,text_to_disp_list,fixation,max_wait_time):
    fixation.draw()
    for td_i, text_to_display in enumerate(text_to_disp_list):
        text_obj.text = text_to_display
        text_obj.pos = (0,(-1*(td_i*2)))
        text_obj.draw()
    win.flip()
    event.waitKeys(maxWait=max_time)
    #nothing to return

def gen_roll_list(Dict_of_Current_Task_Parameters, i_block):
    trial_amb_roll_list = []
    # set up so that first blocks are easier relative to later blocks
    for cur_dict_key in ['easy','med','hard','harder','hardest']:
        # looping through cue conditions... easy -> med -> hard -> hardest
        cur_list_from_dict = Dict_of_Current_Task_Parameters[cur_dict_key] # pulls out the list of trials per block for this condition
        for trl in range(cur_list_from_dict[(i_block)]):
            # adding number of trials for this condtion within this block
            trial_amb_roll_list.append(cur_dict_key)
    random.shuffle(trial_amb_roll_list)
    print(trial_amb_roll_list)
    
    return trial_amb_roll_list

def prep_cue(cue_Params, frac_params, Quantum_Output, i_trial):
    trl_cue = Quantum_Output['cue'][i_trial]
    trl_amb = Quantum_Output['amb'][i_trial]
    rand_prop_select = np.random.randint(0,len(frac_params[trl_cue][trl_amb][0])) # diff conds have diff lengths so grab a num that works for this conds length
    n_red_task = round(frac_params[trl_cue][trl_amb][0][rand_prop_select]*cue_Params['n_dots']) # red first, blue second
    n_blue_task = round(frac_params[trl_cue][trl_amb][1][rand_prop_select]*cue_Params['n_dots']) # red first, blue second
    Quantum_Output['amb_r'][i_trial] = n_red_task/cue_Params['n_dots']
    Quantum_Output['amb_y'][i_trial] = n_blue_task/cue_Params['n_dots']
    
    dots_dotstim_red = visual.DotStim(win, fieldShape='circle', fieldSize=cue_Params['cue_field_size'], speed=10, dotLife=100, nDots=int(n_red_task),  
                                        coherence=0., signalDots='different', noiseDots='walk',  color='red', dotSize=int(cue_Params['dotSize']))
    dots_dotstim_blue = visual.DotStim(win, fieldShape='circle', fieldSize=cue_Params['cue_field_size'], speed=10, dotLife=100, nDots=int(n_blue_task), 
                                        coherence=0., signalDots='different', noiseDots='walk', color='yellow', dotSize=int(cue_Params['dotSize']))

    return dots_dotstim_red, dots_dotstim_blue

def draw_cue(cue_Params, dots_dotstim_red, dots_dotstim_blue, fixation, Quantum_Trl_Durs, frame_rate_def):
    for frame in range(int(Quantum_Trl_Durs['cue']*frame_rate_def)):
        dots_dotstim_red.draw()
        dots_dotstim_blue.draw()
        fixation.draw()
        win.flip()

def draw_ITI(CP_Trl_Durs, frame_rate_def, fixation, fixation_gray):
    for frame in range(int( (frame_rate_def * (CP_Trl_Durs['ITI'][np.random.randint(0,len(CP_Trl_Durs['ITI']))] - 0.75) ))):
        fixation_gray.draw()
        win.flip()
    for frame in range(int(frame_rate_def*0.75)):
        # change back to the black fixation cue for the last 0.75 seconds so they know not to blink
        fixation.draw()
        win.flip()

def draw_initial_fix(Total_Dur, fixation, fixation_gray):
    fixation_gray.draw()
    win.flip()
    core.wait(Total_Dur-1)
    fixation.draw()
    win.flip()
    core.wait(1) # wait 2 seconds before starting to give EEG time to settle

def show_image_get_resp(Dict_of_trl_durs, cur_Img_Dict, fixation, ITI2):
    kb.clearEvents()
    kb.clock.reset()  # restart on each frame to get frame time + button press time
    # display both image house_img, face_img
    skip_check=0
    for frame in range(int(Dict_of_trl_durs['stim']*frame_rate_def)):
        (cur_Img_Dict['Scene_img']).draw()
        (cur_Img_Dict['Face_img']).draw()
        fixation.draw()
        win.flip() # display stimuli
        if skip_check==0:
            keys = kb.getKeys(keyList=resp_keys, waitRelease=False, clear=True)
            if (keys!=[]):
                skip_check=1
    for frame in range( int(round((ITI2*frame_rate_def),0)) ):
        fixation.draw()
        win.flip() # display stimuli
        if skip_check==0:
            keys = kb.getKeys(keyList=resp_keys, waitRelease=False, clear=True)
            if (keys!=[]):
                skip_check=1
    if (keys!=None) and (keys!=[]):
        for key in keys:
            if key.name == resp_keys[0]:
                cur_resp = 0 # [A]
            elif key.name == resp_keys[1]:
                cur_resp = 1 # [S]
            elif key.name == resp_keys[2]:
                cur_resp = 2 # [L]
            elif key.name == resp_keys[3]:
                cur_resp = 3 # [K]
            cur_RT = key.rt # DOUBLE CHECK THIS
    else:
        cur_resp = -1
        cur_RT = -1 # because no response
            
    return cur_RT, cur_resp

def mri_show_image_get_resp(Dict_of_trl_durs, cur_Img_Dict, fixation, ITI2, Time_Since_Run):
    kb.clearEvents()
    kb.clock.reset()  # restart on each frame to get frame time + button press time
    # display both image house_img, face_img
    check_for_press = 1
    for frame in range(int(Dict_of_trl_durs['stim']*frame_rate_def)):
        (cur_Img_Dict['Scene_img']).draw()
        (cur_Img_Dict['Face_img']).draw()
        fixation.draw()
        win.flip() # display stimuli
        if check_for_press==1:
            keys = kb.getKeys(keyList=resp_keys, waitRelease=False, clear=True)
            if (keys!=[]):
                subRespo_T=Time_Since_Run.getTime()
                check_for_press = 0
    for frame in range( int(round((ITI2*frame_rate_def),0)) ):
        fixation.draw()
        win.flip() # display stimuli
        if check_for_press==1:
            keys = kb.getKeys(keyList=resp_keys, waitRelease=False, clear=True)
            if (keys!=[]):
                subRespo_T=Time_Since_Run.getTime()
                check_for_press = 0
    if (keys!=None) and (keys!=[]):
        for key in keys:
            if key.name == resp_keys[0]:
                cur_resp = 0 # [A]
            elif key.name == resp_keys[1]:
                cur_resp = 1 # [S]
            elif key.name == resp_keys[2]:
                cur_resp = 2 # [L]
            elif key.name == resp_keys[3]:
                cur_resp = 3 # [K]
            cur_RT = key.rt # DOUBLE CHECK THIS
    else:
        cur_resp = -1
        cur_RT = -1 # because no response
        subRespo_T=np.nan
            
    return cur_RT, cur_resp, subRespo_T

#### FUNCTIONS FOR PREPPING BLOCK TRIALS
def generate_state(Quantum_Task_Parameters):
    hidden_state = []
    cur_state = np.random.choice([-1,1]) # will be either -1 or 1
    sc_range = np.array(range(Quantum_Task_Parameters['switchRange'][0],Quantum_Task_Parameters['switchRange'][1],1)) # makes an array of 10->30 in steps of 1
    while len(hidden_state) < (Quantum_Task_Parameters['n_trials']*Quantum_Task_Parameters['n_blocks']):
        r = sc_range[np.random.permutation(len(sc_range))]
        for cr in range(r[0]):
            hidden_state.append(cur_state)
        cur_state = cur_state*-1
    hidden_state = np.array(hidden_state).flatten()
    hidden_state = hidden_state[:(Quantum_Task_Parameters['n_trials']*Quantum_Task_Parameters['n_blocks'])]
    
    return hidden_state
    
def load_and_gen_imgs(n_trials, scene_code_list, face_code_list):
    # random sequence of face & scene picture presentation
    Img_Dict = {}
    img_choices = {'Gender':{-1:'female', 1:'male'}, 'Type':{-1:'landscape', 1:'city'}}
    img_list_choices = {'Faces':{-1:glob.glob(os.path.join(os.getcwd(),'images','Faces',('female_[0-9][0-9].jpg'))),  1:glob.glob(os.path.join(os.getcwd(),'images','Faces',('male_[0-9][0-9].jpg')))}, 
                        'Scenes':{-1:glob.glob(os.path.join(os.getcwd(),'images','Scenes',('landscape_[0-9][0-9].jpg'))),  1:glob.glob(os.path.join(os.getcwd(),'images','Scenes',('city_[0-9][0-9].jpg')))}}
    # randomly select pics from list, only load same number of pics as number of trials to save memory
    for i,f in enumerate(np.random.randint(low=0, high=len(img_list_choices['Faces'][-1]), size=n_trials)): 
#        Quantum_Output['face_gender'].append(img_choices['Gender'][face_code_list[i]])
#        Quantum_Output['scene_type'].append(img_choices['Type'][scene_code_list[i]])
#        Quantum_Output['face_img'].append(os.path.basename(img_list_choices['Faces'][face_code_list[i]][f]))
#        Quantum_Output['scene_img'].append(os.path.basename(img_list_choices['Scenes'][scene_code_list[i]][f]))
        Img_Dict[i] = {'Face_img': visual.ImageStim(win=win, image=(Image.open(img_list_choices['Faces'][face_code_list[i]][f])).convert('L'), opacity=0.55), 
                        'Scene_img': visual.ImageStim(win=win, image=(Image.open(img_list_choices['Scenes'][scene_code_list[i]][f])).convert('L'), opacity=0.85)}
    
    return Img_Dict

def prep_Quantum_block_trials(Quantum_Task_Parameters, subInfo, state, i_block):
    Quantum_Output = {'block': np.ones(Quantum_Task_Parameters['n_trials'])*i_block,  'trial': np.zeros(Quantum_Task_Parameters['n_trials']),
                    'face_img': [],  'scene_img': [],   'face_gender': [],  'scene_type': [],
                    'face_code': np.random.choice([1,-1], size=Quantum_Task_Parameters['n_trials']), 
                    'scene_code': np.random.choice([1,-1], size=Quantum_Task_Parameters['n_trials']),
                    'amb': [], 'amb_r': np.zeros(Quantum_Task_Parameters['n_trials']), 'amb_y': np.zeros(Quantum_Task_Parameters['n_trials']),
                    'state': [],  'cue': np.zeros(Quantum_Task_Parameters['n_trials']),  'cue_predominant_color': [],  'target': [],
                    'corr_resp': np.zeros(Quantum_Task_Parameters['n_trials']), 'subj_resp': (np.ones(Quantum_Task_Parameters['n_trials'])*-1),
                    'correct': np.zeros(Quantum_Task_Parameters['n_trials']),   'RT': (np.ones(Quantum_Task_Parameters['n_trials'])*-1)}
    if subInfo['method']=='MRI':
        Quantum_Output['Time_Since_Run_Cue_Prez'] = np.zeros(Quantum_Task_Parameters['n_trials'])
        Quantum_Output['Time_Since_Run_Photo_Prez'] = np.zeros(Quantum_Task_Parameters['n_trials'])
        Quantum_Output['Time_Since_Run_subRespo'] = np.zeros(Quantum_Task_Parameters['n_trials'])
        Quantum_Output['Time_Since_Run_Feedback_Prez'] = np.zeros(Quantum_Task_Parameters['n_trials'])

    # set up trial list for shuffling
    Quantum_Output['amb'] = gen_roll_list(Quantum_Task_Parameters, i_block) # had to modify shuffle call to get current block proportions
    
    ##### State (not explicitely cued AND covers trials not refreshes)
    if state == []:
        Quantum_Output['state'] = generate_state(Quantum_Task_Parameters)
        Quantum_Output['state'] = Quantum_Output['state'][:Quantum_Task_Parameters['n_trials']]
    else:
        Quantum_Output['state'] = state[(Quantum_Task_Parameters['n_trials']*(i_block)):(Quantum_Task_Parameters['n_trials']*(i_block+1))]

    ##### Stimuli Variables
    # random sequence of face & scene picture presentation
    Img_Dict = {}
    img_choices = {'Gender':{-1:'female', 1:'male'}, 'Type':{-1:'landscape', 1:'city'}}
    img_list_choices = {'Faces':{-1:glob.glob(os.path.join(os.getcwd(),'images','Faces',('female_[0-9][0-9].jpg'))),  1:glob.glob(os.path.join(os.getcwd(),'images','Faces',('male_[0-9][0-9].jpg')))}, 
                        'Scenes':{-1:glob.glob(os.path.join(os.getcwd(),'images','Scenes',('landscape_[0-9][0-9].jpg'))),  1:glob.glob(os.path.join(os.getcwd(),'images','Scenes',('city_[0-9][0-9].jpg')))}}
    # randomly select pics from list, only load same number of pics as number of trials to save memory
    for i,f in enumerate(np.random.randint(low=0, high=len(img_list_choices['Faces'][-1]), size=Quantum_Task_Parameters['n_trials'])): #Quantum_Task_Parameters['n_trials'])):
        Quantum_Output['face_gender'].append(img_choices['Gender'][Quantum_Output['face_code'][i]])
        Quantum_Output['scene_type'].append(img_choices['Type'][Quantum_Output['scene_code'][i]])
        Quantum_Output['face_img'].append(os.path.basename(img_list_choices['Faces'][Quantum_Output['face_code'][i]][f]))
        Quantum_Output['scene_img'].append(os.path.basename(img_list_choices['Scenes'][Quantum_Output['scene_code'][i]][f]))
        Img_Dict[i] = {'Face_img': visual.ImageStim(win=win, image=(Image.open(img_list_choices['Faces'][Quantum_Output['face_code'][i]][f])).convert('L'), opacity=0.55), 
                        'Scene_img': visual.ImageStim(win=win, image=(Image.open(img_list_choices['Scenes'][Quantum_Output['scene_code'][i]][f])).convert('L'), opacity=0.85)}

        ###### Create trial lists outside trial loop as well (save time)
        ### Generate trial cue & stimuli.
        Quantum_Output['cue'][i] = np.random.choice([-1,1]) # 1 = blue,  -1 = red.
        if abs( np.sum( Quantum_Output['cue'][int(i-6):i] ) ) == 6:
            Quantum_Output['cue'][i] = Quantum_Output['cue'][i]*-1
            print("6 in a row so manually forcing it to change to the other color to break the streak")
        if Quantum_Output['cue'][i] == 1:
            Quantum_Output['cue_predominant_color'].append('yellow')
        else:
            Quantum_Output['cue_predominant_color'].append('red')
        
        ### Record correct choice
        # if state*cue = 1, focusing on scene ... if state*cue = -1, focusing on face
        if (Quantum_Output['state'][i] * Quantum_Output['cue'][i]) == 1:
            Quantum_Output['target'].append("scene")
        else:
            Quantum_Output['target'].append("face")
        # figure out what correct answer would be given state and cue and target
        # will be 4 answer choices for the 4 combinations of -1 and 1
        if (Quantum_Output['state'][i] * Quantum_Output['cue'][i]) == 1:
            # SCENE is target ... check scene image (city or landscape)
            if Quantum_Output['scene_code'][i] == 1: # city
                Quantum_Output['corr_resp'][i] = 3 #resp_keys[3]
            else:
                Quantum_Output['corr_resp'][i] = 2 #resp_keys[2]
        elif (Quantum_Output['state'][i] * Quantum_Output['cue'][i]) == -1:
            # FACE is target ... check face image (male or female)
            if Quantum_Output['face_code'][i] == 1: # male
                Quantum_Output['corr_resp'][i] = 1 #resp_keys[1]
            else:
                Quantum_Output['corr_resp'][i] = 0 #resp_keys[0]

    return Quantum_Output, Img_Dict

def prep_CP_block_trials(CP_Task_Parameters, i_block):
    CP_Output = {'block': np.zeros(CP_Task_Parameters['n_trials']),    'trial': np.zeros(CP_Task_Parameters['n_trials']), 
                'amb': [], 'amb_r': np.zeros(CP_Task_Parameters['n_trials']), 'amb_y': np.zeros(CP_Task_Parameters['n_trials']),
                'cue': np.zeros(CP_Task_Parameters['n_trials']),       'cue_predominant_color': [],        'corr_resp': None,   
                'subj_resp': (np.ones(CP_Task_Parameters['n_trials'])*-1),  'correct': np.zeros(CP_Task_Parameters['n_trials']),  'RT': (np.ones(CP_Task_Parameters['n_trials'])*-1)}
    
    # set up trial list for shuffling
    CP_Output['amb'] = gen_roll_list(CP_Task_Parameters, i_block) # had to modify shuffle call to get current block proportions
    
    for i in range(CP_Task_Parameters['n_trials']):
        ### Generate trial cue & stimuli.
        CP_Output['cue'][i] = np.random.choice([-1,1]) # 1 = blue,  -1 = red.
        if abs( np.sum( CP_Output['cue'][int(i-6):i] ) ) == 6:
            CP_Output['cue'][i] = CP_Output['cue'][i]*-1
            print("6 in a row so manually forcing it to change to the other color to break the streak")
        if CP_Output['cue'][i] == 1:
            CP_Output['cue_predominant_color'].append('yellow')
        else:
            CP_Output['cue_predominant_color'].append('red')
    CP_Output['corr_resp'] = CP_Output['cue']+1

    return CP_Output


# LOOP THROUGH THE TASKS
if subInfo['method']!="MRI":
    # load trial figure to display in instructions AND THEN load task images (stimuli)
    trial_fig = visual.ImageStim(win=win, image=os.path.join(os.getcwd(),'images','Trial_Figure_MRI.png'), size=(41,19), pos=(0,-1.45))
    Img_Dict = load_and_gen_imgs(5, np.random.choice([1,-1], size=5), np.random.choice([1,-1], size=5)) # set up a smaller Img_Dict for the instruction screens
    
    #################################################################################################################################
    ####################################################### Give Instruction ########################################################
    repeat_instructions = True
    while repeat_instructions:
        if subInfo['current_task']=="Quantum":
            Quantum_Instructions.draw()
            trial_fig.draw()
            win.flip()
            event.waitKeys()
            give_example(["CUE SCREEN\n",  "DELAY SCREEN\n\n\nWere there more red or yellow dots?\nThis determines what task you perform\n(scene or face)",
                        "\n\n  TASK SCREEN\n\nIf face task, female or male?\nIf scene task, city or landscape?"],
                        ["cue","delay","task"], {'cue':2, 'delay':9, 'task':15}, frac_params, cue_Params, fixation)
            give_example(["Note that the actual task will move at a faster pace\n\nPress any key to see an example of a full trial at real pace",
                        "CUE SCREEN", "DELAY SCREEN\n\n", "\n\n\n\nTASK SCREEN", "DELAY SCREEN\n\n", "\n\n\n\nFEEDBACK"], ["text","cue","delay","task","delay","feedback"],
                        {'text':10, 'cue':Quantum_Trl_Durs['cue'], 'delay':3.5, 'task':Quantum_Trl_Durs['stim'], 'delay':2.5, 'feedback':Quantum_Trl_Durs['feedback']}, frac_params, cue_Params, fixation)
        else:
            give_example(["Task instruction: you will see a circle made up of random dots. You need to report what color (red OR yellow) was more common in the random dots.\n\nPress any key to see a slowed down example of a task trial.",
                            "CUE SCREEN\n\n",  "\n\n\n\n\n\n\t\t\tResponse SCREEN\n\nPress the [R] key if more red and the [Y] key if more yellow"], 
                            ["text","cue","delay"], {'text':15,'cue':2.5, 'delay':6}, frac_params, cue_Params, fixation)
            give_example(["Note that the actual task will move at a faster pace\n\nPress any key to see an example of a trial at real pace",
                            "CUE SCREEN", "\n\n\n\n\nRESPONSE SCREEN"],  ["text","cue","delay"],  {'text':15, 'cue':CP_Trl_Durs['cue'], 'delay':CP_Trl_Durs['resp']}, frac_params, cue_Params, fixation)
        # check if they want to repeat instructions or move onto task
        repeat_instructions, do_task = check_prac_task_choice(CheckChoice_Text,repeat_instructions,0)

    event.clearEvents()

    ###########################################################################################################################
    ###################################################### Run trials #########################################################
    if subInfo['method'] == 'EEG':
        ##### TTL Pulse trigger
        port.open()
    
    if subInfo['current_task']=="Quantum":
        while do_task == 0:
            # Prep block trials
            prac_dict, Img_Dict = prep_Quantum_block_trials(Quantum_Task_Parameters, subInfo, [], 0)

            if subInfo['method'] == 'EEG':
                port.write(trigDict['startSaveflag'])
                draw_initial_fix(3, fixation, fixation_gray)

            # RUN PRACTICE BLOCK ... get trials ready
            cur_block_acc = 0
            
            for i_trial in range(0,Quantum_Task_Parameters['n_prac_trials']):
                ##### Define cue stimuli (color distribution of dots) based on trial_cue and trial_amb.
                dots_dotstim_red, dots_dotstim_blue = prep_cue(cue_Params, frac_params, prac_dict, i_trial)
                
                #################################################################################################################### Actually generating stimuli on window.
                give_intertrial_text(text,["Press any key to start the next trial.\n\n\n\n\n\n\n", "\n\n\n\n\nRESPONSE REMINDER",
                                "\n\n\n\n\n\n[A] = Female,  [S] = Male\t\t\t\t\t\t\t\t\t\t[K] = City,  [L] = Landscape"], fixation_gray, max_time)
                
                ### Cue (choice 1): Generate the 2 colors separately, using dotstim
                if subInfo['method'] == 'EEG':
                    win.callOnFlip(port.write,bytes([ trigDict['state'][prac_dict['state'][i_trial]][prac_dict['cue'][i_trial]][prac_dict['amb'][i_trial]] ]))
                draw_cue(cue_Params, dots_dotstim_red, dots_dotstim_blue, fixation, Quantum_Trl_Durs, frame_rate_def)
                
                ### Delay period
                if subInfo['method'] == 'EEG':
                    win.callOnFlip(port.write,bytes([ trigDict['delay'] ]))
                    for frame in range(int(Quantum_Trl_Durs['delay']*frame_rate_def)):
                        draw_shapes([fixation])
                else:
                    for frame in range(int(np.random.randint(1,9)*Quantum_Trl_Durs['delay']*frame_rate_def)):
                        draw_shapes([fixation])
                
                ### Stim + Response window
                if subInfo['method'] == 'EEG':
                    win.callOnFlip(port.write,bytes([ trigDict['target'][prac_dict['target'][i_trial]][prac_dict['face_gender'][i_trial]][prac_dict['scene_type'][i_trial]] ]))
                prac_dict['RT'][i_trial], prac_dict['subj_resp'][i_trial] = show_image_get_resp(Quantum_Trl_Durs, Img_Dict[(i_trial)], fixation, int(np.random.randint(1,6)))
                
                if subInfo['method'] == 'EEG':
                    win.callOnFlip(port.write,bytes([ trigDict['resp'][prac_dict['subj_resp'][i_trial]] ]))
                ## record accuracy on current trial
                if ( prac_dict['corr_resp'][i_trial] == prac_dict['subj_resp'][i_trial] ):
                    prac_dict['correct'][i_trial] = 1
                    cur_block_acc += 1
                #if testing_for_MRI and subInfo['method']=='Behavioral':
                #    for frame in range(int(np.random.randint(1,9)*Quantum_Trl_Durs['delay']*frame_rate_def)):
                #        draw_shapes([fixation])
                
                ### Feedback window
                for frame in range(int(Quantum_Trl_Durs['feedback']*frame_rate_def)):
                    horz_line.draw()
                    if prac_dict['correct'][i_trial]==1:
                        vert_line.draw() # add verticle line to make plus sign feedback
                    win.flip()
                print("\ttrial:", i_trial, "\tstate:", prac_dict['state'][i_trial], "\tamb:", prac_dict['amb'][i_trial], "\tgender:", prac_dict['face_gender'][(i_trial)], "\tstyle:", prac_dict['scene_type'][(i_trial)], "\tsubj resp:", prac_dict['subj_resp'][i_trial], "\tcorr resp:", prac_dict['corr_resp'][i_trial], "\tfeedback:",prac_dict['correct'][i_trial])
                if subInfo['method'] == 'EEG':
                    win.callOnFlip(port.write,bytes([ trigDict['feedback_ITI'][prac_dict['correct'][i_trial]] ])) # to be called once ITI text pops up for next trial
                
            ##### TTL Pulse trigger
            if subInfo['method'] == 'EEG':
                core.wait(1)
                port.write(trigDict['stopSaveflag'])

            # save out practice block info
            prac_output_file = pd.DataFrame(prac_dict)
            prac_output_file.to_csv(os.path.join(output_dir, ("Data_"+subInfo['method']), ("sub-" + str(int(subInfo['Subject_ID'])) + "_task-QuantumPractice_" + str(datetime.today().strftime('%m-%d-%y')) + ".csv")), index=False)
            
            ### block screens
            cur_block_acc = round(((cur_block_acc/Quantum_Task_Parameters['n_prac_trials'])*100),2)
            disp_block_text(EndofBlock_Text, cur_block_acc, False, max_time)
            # check if we need to redo practice
            CheckChoice_Text.text="Press [P] to start the practice or\npress [Y] to start the task."
            repeat_instructions, do_task = check_prac_task_choice(CheckChoice_Text,repeat_instructions, do_task)
            
        # ACTUAL TASK
        if do_task==1:
            # Prep block trials
            state = generate_state(Quantum_Task_Parameters)
            
            for i_block in range(0,Quantum_Task_Parameters['n_blocks']):
                Quantum_Output, Img_Dict = prep_Quantum_block_trials(Quantum_Task_Parameters, subInfo, state, i_block)
                
                if subInfo['method']=='EEG':
                    ##### TTL Pulse trigger
                    port.write(trigDict['startSaveflag']) # start saving at the start of the block
                    
                draw_initial_fix(3, fixation, fixation_gray)
                cur_block_acc = 0
                
                for i_trial in range(0,Quantum_Task_Parameters['n_trials']):
                    ##### Generate trial cue & stimuli.
                    if testing_for_MRI and subInfo['method']=='Behavioral':
                        ITI = make_ITI(ITIs_for_runs[i_block], i_trial)
                        ITI2 = make_ITI(ITIs_for_runs[i_block], (i_trial+Quantum_Task_Parameters['n_trials']))
                        ITI3 = make_ITI(ITIs_for_runs[i_block], (i_trial+(2*Quantum_Task_Parameters['n_trials']))) # iti delay
                    Quantum_Output['trial'][i_trial] = i_trial+1
                    
                    ##### Define cue stimuli (color distribution of dots) based on trial_cue and trial_amb.
                    dots_dotstim_red, dots_dotstim_blue = prep_cue(cue_Params, frac_params, Quantum_Output, i_trial)
                    
                    #################################################################################################################### Actually generating stimuli on window.
                    ### Cue (choice 1): Generate the 2 colors separately, using dotstim
                    if subInfo['method']=='EEG':
                        win.callOnFlip(port.write,bytes([ trigDict['state'][Quantum_Output['state'][i_trial]][Quantum_Output['cue'][i_trial]][Quantum_Output['amb'][i_trial]] ]))
                    draw_cue(cue_Params, dots_dotstim_red, dots_dotstim_blue, fixation, Quantum_Trl_Durs, frame_rate_def)
                    
                    ### Delay period
                    if subInfo['method']=='EEG':
                        win.callOnFlip(port.write,bytes([ trigDict['delay'] ]))
                    if testing_for_MRI and subInfo['method']=='Behavioral':
                        for frame in range(int(ITI*frame_rate_def)):
                            fixation.draw()
                            win.flip()
                    else:
                        for frame in range(int(Quantum_Trl_Durs['delay']*frame_rate_def)):
                            draw_shapes([fixation])
                    
                    ### Stim + Response window
                    if subInfo['method']=='EEG':
                        win.callOnFlip(port.write,bytes([ trigDict['target'][Quantum_Output['target'][i_trial]][Quantum_Output['face_gender'][i_trial]][Quantum_Output['scene_type'][i_trial]] ]))
                    Quantum_Output['RT'][i_trial], Quantum_Output['subj_resp'][i_trial] = show_image_get_resp(Quantum_Trl_Durs, Img_Dict[(i_trial)], fixation, ITI2)
                    if subInfo['method']=='EEG':
                        win.callOnFlip(port.write,bytes([ trigDict['resp'][Quantum_Output['subj_resp'][i_trial]] ]))
                        win.flip()
                    # get correctness
                    Quantum_Output['correct'][i_trial] = (Quantum_Output['corr_resp'][i_trial]==Quantum_Output['subj_resp'][i_trial])
                    if Quantum_Output['correct'][i_trial]==1:
                        cur_block_acc += 1
                    
                    ### Post-stim and Pre-feedback iti
                    #if testing_for_MRI and subInfo['method']=='Behavioral':
                    #    fixation.draw()
                    #    win.flip()
                    #    core.wait(ITI2)
                    
                    ### Feedback window
                    for frame in range( int(Quantum_Trl_Durs['feedback']*frame_rate_def)-1 ):
                        horz_line.draw()
                        if Quantum_Output['correct'][i_trial]==1:
                            vert_line.draw() # add verticle line to make plus sign feedback
                        win.flip()
                    
                    ### ITI window
                    # send trigger and then draw object
                    print("Block:", (i_block+1), "\tTrial:", i_trial, "\tState:", Quantum_Output['state'][i_trial], "\tAmb:", Quantum_Output['amb'][i_trial], "\tGender:",  Quantum_Output['face_gender'][i_trial], "\tType:", Quantum_Output['scene_type'][i_trial], "\tsubj resp:", Quantum_Output['subj_resp'][i_trial], "\tcorr resp:", Quantum_Output['corr_resp'][i_trial], "\tfeedback:", Quantum_Output['correct'][i_trial])
                    if subInfo['method']=='EEG':
                        win.callOnFlip(port.write,bytes([ trigDict['feedback_ITI'][Quantum_Output['correct'][i_trial]] ]))
                    if testing_for_MRI and subInfo['method']=='Behavioral':
                        fixation.draw()
                        win.flip()
                        core.wait(ITI3)
                    else:
                        draw_ITI(Quantum_Trl_Durs, frame_rate_def, fixation, fixation_gray)

                ### block screens
                # save out data from last block
                Quantum_Output_df = pd.DataFrame(Quantum_Output)
                Quantum_Output_df.to_csv(os.path.join(output_dir, ("Data_" + subInfo['method']), ("sub-" + str(int(subInfo['Subject_ID'])) + "_task-" + subInfo['current_task'] + "_Block_00" + str(int(i_block+1)) + ".csv")), index=False)
                if subInfo['method']=='EEG':
                    ##### TTL Pulse trigger
                    core.wait(1)
                    port.write(trigDict['stopSaveflag'])
                cur_block_acc = round(100*(cur_block_acc/Quantum_Task_Parameters['n_trials']), 2)
                disp_block_text(EndofBlock_Text, cur_block_acc,((i_block+1)==Quantum_Task_Parameters['n_blocks']), max_time)
                
    else:

        for i_block in range(0,CP_Task_Parameters['n_blocks']):
            # Prep trials
            CP_Output = prep_CP_block_trials(CP_Task_Parameters, i_block)

            if subInfo['method']=='EEG':
                ##### TTL Pulse trigger
                port.write(trigDict['startSaveflag'])
            draw_initial_fix(3, fixation, fixation_gray)
            
            for i_trial in range(0,CP_Task_Parameters['n_trials']):
                ##### Generate trial cue & stimuli.
                CP_Output['block'][i_trial] = i_block+1
                CP_Output['trial'][i_trial] = i_trial+1

                ##### Define cue stimuli (color distribution of dots) based on trial_cue and trial_amb.
                dots_dotstim_red, dots_dotstim_blue = prep_cue(cue_Params, frac_params, CP_Output, i_trial)

                ### Cue (choice 1): Generate the 2 colors separately, using dotstim
                if subInfo['method']=='EEG':
                    win.callOnFlip(port.write,bytes([ trigDict['state'][1][CP_Output['cue'][i_trial]][CP_Output['amb'][i_trial]] ]))
                draw_cue(cue_Params, dots_dotstim_red, dots_dotstim_blue, fixation, CP_Trl_Durs, frame_rate_def)
                
                ### Get Resp
                kb.clearEvents()
                kb.clock.reset()  # restart on each frame to get frame time + button press time
                for frame in range(int(CP_Trl_Durs['resp']*frame_rate_def)):
                    draw_shapes([fixation])
                    keys = kb.getKeys(keyList=['r','y'], waitRelease=False, clear=True) # 1 = blue,  -1 = red.
                    if (keys!=None) and (keys!=[]):
                        # send trigger and then save out resp info
                        for key in keys:
                            if subInfo['method']=='EEG':
                                win.callOnFlip(port.write,bytes([ trigDict['resp'][key.name] ]))
                            if key.name == 'r':
                                CP_Output['subj_resp'][i_trial] = 0
                            elif key.name == 'y':
                                CP_Output['subj_resp'][i_trial] = 2
                            CP_Output['RT'][i_trial] = key.rt # DOUBLE CHECK THIS
                        win.flip()
                        break

                ## record accuracy on current trial
                CP_Output['correct'][i_trial] = (CP_Output['corr_resp'][i_trial]==CP_Output['subj_resp'][i_trial])
                print("block:", (i_block+1), "\ttrial:", (i_trial+1), "\tamb:", CP_Output['amb'][i_trial], "\tsubj resp:", CP_Output['subj_resp'][i_trial], "\tcorr resp:", CP_Output['corr_resp'][i_trial])
                
                ### ITI window
                if subInfo['method']=='EEG':
                    win.callOnFlip(port.write,bytes([ trigDict['feedback_ITI'][CP_Output['correct'][i_trial]] ]))
                draw_ITI(CP_Trl_Durs, frame_rate_def, fixation, fixation_gray)

            ### Block Screen
            # save out data from last block
            CP_Output_df = pd.DataFrame(CP_Output)
            CP_Output_df.to_csv(os.path.join(output_dir, ("Data_" + subInfo['method']), ("sub-" + str(int(subInfo['Subject_ID'])) + "_task-" + subInfo['current_task'] + "_Block_00" + str(int(i_block+1)) + ".csv")), index=False)
            if subInfo['method']=='EEG':
                ##### TTL Pulse trigger
                core.wait(1)
                port.write(trigDict['stopSaveflag'])
            disp_block_text(EndofBlock_Text, 0, ((i_block+1)==CP_Task_Parameters['n_blocks']), max_time)

    ##### Close window and port (if EEG)
    win.close() # close visual window
    if subInfo['method']=='EEG':
        port.close()





elif subInfo['method']=="MRI":
    ###########################################################################################################################
    ###################################################### Run trials #########################################################
    state = generate_state(Quantum_Task_Parameters)
    
    Welc.draw()
    win.flip()
    event.waitKeys(maxWait=3)
    
    startlist = [resp_keys[0], resp_keys[1], resp_keys[2], resp_keys[3], '1']
    
    block_start = int(subInfo['Block']) - 1
    for i_block in range(block_start, Quantum_Task_Parameters['n_blocks']):
        cur_datetime = datetime.today().strftime('%Y%m%d_%I%M')
        # Prep block trials
        Quantum_Output, Img_Dict = prep_Quantum_block_trials(Quantum_Task_Parameters, subInfo, state, i_block)

        Directions.draw() # have them press when they're ready
        win.flip()
        event.waitKeys(keyList=startlist) # only use subj resp keys so scanner can prep in background
        
        ##### TTL Pulse trigger
        Wait_for_Scanner.draw()
        win.flip()
        key = event.waitKeys(keyList=['lshift','z','equal'])
        #print(key[0])
        
        #### Setting up a global clock to track initiation of experiment to end
        Time_Since_Run = core.MonotonicClock()  # to track the time since experiment started, this way it is very flexible compared to psychopy.clock
        ##### 2 seconds Intial fixation
        fixation.draw()
        win.flip()
        core.wait(6)
        cur_block_acc = 0

        for i_trial in range(0,Quantum_Task_Parameters['n_trials']):
            ##### Generate trial cue & stimuli.
            ITI = make_ITI(ITIs_for_runs[i_block], (i_trial*3)) # delay between cue and stim
            ITI2 = make_ITI(ITIs_for_runs[i_block], ((i_trial*3)+1)) # delay between stim and feedback
            ITI3 = make_ITI(ITIs_for_runs[i_block], ((i_trial*3)+2)) # iti delay
            print('iti1 ind:',ITI,'\titi2 ind:', ITI2, '\titi3 ind:', ITI3)
            Quantum_Output['trial'][i_trial] = i_trial+1
            
            ##### Define cue stimuli (color distribution of dots) based on trial_cue and trial_amb.
            dots_dotstim_red, dots_dotstim_blue = prep_cue(cue_Params, frac_params, Quantum_Output, i_trial)
            
            #################################################################################################################### Actually generating stimuli on window.
            ### Cue (choice 1): Generate the 2 colors separately, using dotstim
            Quantum_Output['Time_Since_Run_Cue_Prez'][i_trial]=Time_Since_Run.getTime()
            draw_cue(cue_Params, dots_dotstim_red, dots_dotstim_blue, fixation, Quantum_Trl_Durs, frame_rate_def)
            
            ### Delay period
            #for frame in range( int(ITI*frame_rate_def) ):
            #    fixation.draw()
            #    win.flip()
            fixation.draw()
            win.flip()
            core.wait(ITI)
            
            ### Stim + Response window
            Quantum_Output['Time_Since_Run_Photo_Prez'][i_trial]=Time_Since_Run.getTime()
            Quantum_Output['RT'][i_trial], Quantum_Output['subj_resp'][i_trial], Quantum_Output['Time_Since_Run_subRespo'][i_trial] = mri_show_image_get_resp(Quantum_Trl_Durs, Img_Dict[(i_trial)], fixation, ITI2, Time_Since_Run)
            # get correctness
            Quantum_Output['correct'][i_trial] = (Quantum_Output['corr_resp'][i_trial]==Quantum_Output['subj_resp'][i_trial])
            if Quantum_Output['correct'][i_trial]==1:
                cur_block_acc += 1
            
            ### Post-stim and Pre-feedback iti
            #fixation.draw()
            #win.flip()
            #core.wait(ITI2)
            
            ### Feedback window
            Quantum_Output['Time_Since_Run_Feedback_Prez'][i_trial] = Time_Since_Run.getTime()
            for frame in range( int(Quantum_Trl_Durs['feedback']*frame_rate_def)-1 ):
                horz_line.draw()
                if Quantum_Output['correct'][i_trial]==1:
                    vert_line.draw() # add verticle line to make plus sign feedback
                win.flip()
            
            ### ITI window
            #for frame in range( int(ITI3*frame_rate_def) ):
            #    fixation.draw()
            #    win.flip() 
            fixation.draw()
            win.flip()
            core.wait(ITI3)
            print("Block:", i_block, "\tTrial:", i_trial, "\tState:", Quantum_Output['state'][i_trial], "\tAmb:", Quantum_Output['amb'][i_trial], "\tGender:",  Quantum_Output['face_gender'][i_trial], "\tType:", Quantum_Output['scene_type'][i_trial], "\tsubj resp:", Quantum_Output['subj_resp'][i_trial], "\tcorr resp:", Quantum_Output['corr_resp'][i_trial], "\tfeedback:", Quantum_Output['correct'][i_trial])
            
            start_save = datetime.now()
            # save out data from last block
            if ((i_trial+1)%5 == 0) or ((i_trial+1)==Quantum_Task_Parameters['n_trials']):
                # Quantum_Output_df = pd.DataFrame(Quantum_Output)
                # Quantum_Output_df.to_csv(os.path.join(output_dir, ("Data_" + subInfo['method']), ("sub-" + str(int(subInfo['Subject_ID'])) + "_task-" + subInfo['current_task'] + "_Block_00" + str(int(i_block+1)) + ".csv")), index=False)
                with open(os.path.join(output_dir, ("Data_" + subInfo['method']), ("sub-" + str(int(subInfo['Subject_ID'])) + "_task-" + subInfo['current_task'] + "_session-00" + str(subInfo['Session']) + "_Block_00" + str(int(i_block+1)) + "_date-" + cur_datetime + ".csv")), 'w', newline='') as f_output:
                    csv_output = csv.writer(f_output)
                    csv_output.writerow(Quantum_Output.keys())
                    csv_output.writerows([*zip(*Quantum_Output.values())])
                print('time to save: ',(datetime.now()-start_save), '\ttime so far: ', Time_Since_Run.getTime())
            
        ### block screens
        core.wait(3) # add another blank screen for 3 seconds before ending the block
        print("Onset of block screen: ", Time_Since_Run.getTime(),"\n\n\n\n")
        EndofBlock_Text.text = "Block " + str(int(i_block+1)) + " finished.\n\nAccuracy was " + str(int(100*(cur_block_acc/Quantum_Task_Parameters['n_trials']))) + " percent\n\nPrepping next task block now...Please stay still"
        EndofBlock_Text.draw()
        win.flip()
        core.wait(5) # give myself enough time to turn off the triggers
        win.winHandle.minimize() # temporarily minimize the window
        win.winHandle.set_fullscreen(False)
        win.flip()
        dlg_cont = gui.DlgFromDict(dictionary={'Click OK when ready to move on':''}, title="Click okay when ready to move on", sortKeys=False)
        win.winHandle.maximize()
        win.winHandle.set_fullscreen(True) 
        win.winHandle.activate()
        win.flip()
        core.wait(0.5)
        

    ##### Close window
    win.close() # close visual window
