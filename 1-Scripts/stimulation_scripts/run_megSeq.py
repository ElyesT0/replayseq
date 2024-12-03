"""
Title: MegSeq
Experiment: MEG

Author: Fosca Al Roumi, Elyès Tabbane
Date: 04-07-2024

Version: 1.1.2

Description:
    Delayed sequence reproduction. 9 sequence structures are presented. Each sequence structure is presented 30 times over 2 blocks (15 times each).
    In between each block, there is a resting state phase of 1min30. Participant is asked to close their eyes and rest.
    Goal is observing replay during maintenance phase (6 seconds of iddle fixation after the presentation of a sequence and before the reproduction starts),
    and during the resting period.

Associated environment: megSeq

Dependencies:
    - response_funcs.py : contains the methods enabling the interaction and communication with the MEG machine

Instructions:
    (1) pyenv version 
    => List the available environments on the machine
    (2) pyenv activate [name of environment]


Notes:
    - Set MEG = False, to test the code on a computer without involving the MEG machine's buttons,
    - set debugging=True, to test the code quickly without long resting phase and breaks,
    - Once resting state is over, save the run. Push keyboard 'n' to end the resting state and start next block.
    

Description of stimuli:
- Hexagonal figure. Six points, 3 by 3 symetry with a vertical axis. 
- Codes for positions of points (they're on a circle so we will use clock time positions to indicate):
    - 10pm: 0,
    - 1pm: 1,
    - 3pm: 2,
    - 5pm: 3,
    - 7pm: 4,
    - 9pm: 5. 
    => In the triggers and in the data we add +1, so it goes from 1 to 6.
"""


# ************************************************************************************************************************************
# Import necessary libraries
# ************************************************************************************************************************************

import numpy as np
from expyriment import design, control, stimuli, misc
import math
import jellyfish
from math import pi as Pi
import random
from datetime import datetime
import time
import response_funcs


debugging=False
MEG = False

if debugging:
    control.set_develop_mode(on=True)
    control.defaults.window_mode = True
else:
    control.defaults.window_mode = False

# ************************************************************************************************************************************
# File paths
# ************************************************************************************************************************************
# -- Directory to save behavioral data
save_dir='data/behavior_output/'
# Prefix for file name
file_name='megSeq_behavioral_'
# Get date for file name
current_date = datetime.now().strftime("%Y-%m-%d")
# Create the output file name using the current date    
output_file = f"output_data_{current_date}.csv"

# -- Directory for deedback sounds
POSITIVE_FEEDBACK = './auditory_feedback/pos1.wav'
NEGATIVE_FEEDBACK = './auditory_feedback/neg1.wav'

# -- Parallel Ports addresses for response buttons in MEG
port_0,port_1,port_2="/dev/parport0","/dev/parport1","/dev/parport2"

# ************************************************************************************************************************************
# Experimental Variables
# ************************************************************************************************************************************

# -- Visual parameters
size_circle_neutral= 10
size_circle_active = 30
size_hexagon = 100
size_container=500 # Size of the container for the hexagon
size_fixation=20 # Both width and height
thickness_fixation=4 #Define fixation linewidth
size_photodiode=100
origin_position=(-300,0) #Define center position for the figure
size_text=30


# -- Timing parameters
SOS = 200
SOA = 400
ITI=1000
duration_fixation = 1000
duration_feedback = 500
duration_break=6000 #time in between end of presentation and start of response phase (indicated by blue cross)
duration_maximumAllowed = 12000
duration_resting=60000
duration_pre_start=1000 # Time before first sequence is shown once the resting state is over
duration_padding=2000 # Standard duration of iddle time during resting phase
duration_trigger=10 # minimum duration before switching back the trigger channel to 0

# -- Experiment structure parameters
nb_trials_per_sequence=15
nb_block_per_sequence=2
sequence_length=12


if debugging:
    ITI=300
    
    #duration_fixation=200
    duration_maximumAllowed=6000
    duration_break=200
    duration_resting=600
    nb_trials_per_sequence=2
    duration_pre_start=1000
    duration_padding=500

# Used to prevent accidental double tap of buttons
last_click=time.time()*1000

    
# ************************************************************************************************************************************
# Experimental Objects
# ************************************************************************************************************************************

# -- Build the Trigger dictionnary
positions=[i for i in range(1,7)]
sequences_labels=['Rep2','CRep2','Rep3', 'CRep3', 'Rep4', 'CRep4','RepEmbed', 'C1RepEmbed', 'C2RepEmbed']

# Case: Debugging (we want less sequences)
if debugging:
    sequences_labels=['Rep2','CRep2','Rep3']

all_codes=['fixation_blue','fixation']

for name in sequences_labels:
    for pos in positions: 
        all_codes.append(f'{name}-{pos}')

all_codes.append('win')
all_codes.append('loss')

Trigger_codes=dict(zip(all_codes,range(4,len(all_codes)+7)))
reverse_Trigger_codes={value: key for key, value in Trigger_codes.items()}

# -- Sequences Tested as a dictionnary
sequence_dict = {'Rep2': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 'CRep2': [0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1],
        'Rep3': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], 'CRep3': [0, 1, 2, 0, 2, 1, 1, 2, 0, 1, 0, 2],
        'Rep4': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], 'CRep4': [0, 1, 2, 3, 2, 1, 3, 0, 0, 1, 3, 2],
        'RepEmbed': [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2], 'C1RepEmbed': [0, 0, 1, 1, 2, 2, 1, 1, 0, 0, 2, 2],
        'C2RepEmbed': [0, 1, 2, 1, 0, 2, 0, 1, 2, 1, 0, 2]}

# Case : Debugging
if debugging:
    sequence_dict = {'Rep2': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 'CRep2': [0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1],
        'Rep3': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]}
    
nb_sequences_tested=len(sequence_dict)


# -- Sequences as an array to be read by Expyriment later on
sequence_tested=[]
# Shuffle sequence names
shuffled_seq_names=[name for name in sequence_dict.keys()]
random.shuffle(shuffled_seq_names)
# We have two blocks per sequences
experiment_seq_names = [name for name in shuffled_seq_names for _ in range(nb_block_per_sequence)]
# Now we get the expressions and we randomize them for each block
experiment_seq_expression=[]
for name in experiment_seq_names:
    block=[]
    # -- Generate all the mappings for the trials
    for k in range(nb_trials_per_sequence):
        # Generate a random mapping applied to the positions on screen. Between 0 and 5
        random_mapping=[i for i in range(6)]
        random.shuffle(random_mapping)
        # Apply the mapping to a sequence and append it to the block
        block.append([random_mapping[i] for i in sequence_dict[name]])
    experiment_seq_expression.append(block)

# -- Response buttons
response_mappings_PC={'i':1, # position 0
                       'u':1, # position 0
                       'j':2, # position 1
                       'k':2, # position 1
                       ',':3, # position 2
                       'c':4, # position 3
                       'x':4, # position 3
                       'f':5, # position 4
                       'd':5, # position 4
                       'r':0, # position 5
                       'e':0, # position 5
        }

response_mappings_MEG={'LY':0, # position 0
                       'RY':1, # position 1
                       'RG':2, # position 2
                       'RR':3, # position 3
                       'LR':4,#position 4
                       'LG':5 # position 5
}


# ************************************************************************************************************************************
# Build the stimuli
# ************************************************************************************************************************************
# -- Create the experiment
# We create the expyriment object here to be able to use the graphical capacities of the library
exp = design.Experiment(name="MegSeq", text_size=30)
# Prepare data output
exp.data_variable_names = ["block", "sequenceName", "trial", "PresentedSequence", "ProducedSequence", "RTs", "Performance"]
# Open the experiment
control.initialize(exp)

response_mappings = response_mappings_PC
if MEG:
    response_mappings = response_mappings_MEG
    response_MEG = response_funcs.response_in_MEG(exp,port_0,port_1,port_2)
    port2send=response_MEG.port2
    port2send.send(data=0)
list_btn=[i for i in response_mappings.keys()]

# -- Fixation cross
# Standard
fixation=stimuli.FixCross(size=(size_fixation,size_fixation),line_width=thickness_fixation)
# Blue (for response screen)
fixation_blue=stimuli.FixCross(size=(size_fixation, size_fixation), line_width=thickness_fixation, colour=(0, 0.5 * 255, 0.5 * 255))


# -- Photodiode item
photodiode = stimuli.Rectangle(size=(size_photodiode,size_photodiode), position=(700,-400), colour=(255,255,255))

# Empty screen
blankscreen = stimuli.Canvas((size_container,size_container),position=origin_position)
fixation.plot(blankscreen)

# -- Screen: Hexagon Standard
# Step 1 : Create the container for the Hexagon standard
container_figure=stimuli.Canvas((size_container,size_container),position=origin_position)

# Step 2: Create the circles elements + Place them on the container_figure
for k in range(6):
    circle_element=stimuli.Circle(radius=size_circle_neutral, 
                                  colour=(0.5*255,0.5*255,0.5*255),
                                  position=(size_hexagon*math.cos(4*Pi/6 - k * Pi/3),
                                          size_hexagon*math.sin(4*Pi/6 - k * Pi/3)))
    circle_element.plot(container_figure)
    
# Step 3 : Place the fixation cross
fixation.plot(container_figure)

# -- Screen: Hexagon Response (with blue fixation)
# We do the same for the response screen. 
container_figure_response=stimuli.Canvas((size_container,size_container),position=origin_position)
for k in range(6):
    circle_element=stimuli.Circle(radius=size_circle_neutral, 
                                  colour=(0.5*255,0.5*255,0.5*255),
                                  position=(size_hexagon*math.cos(4*Pi/6 - k * Pi/3),
                                          size_hexagon*math.sin(4*Pi/6 - k * Pi/3)))
    circle_element.plot(container_figure_response)
fixation_blue.plot(container_figure_response)


# -- Define an Object to interact with the circles. Used to flash the points during presentation or response.
circles=[]
for k in range(6):
    position_circle_element=(size_hexagon*math.cos(4*Pi/6 - k * Pi/3),size_hexagon*math.sin(4*Pi/6 - k * Pi/3))
    circle_element=stimuli.Circle(radius=size_circle_active, # This value changed
                                  colour=(255,255,255),
                                  position=tuple(x+y for x,y in zip (origin_position,position_circle_element)))
    circles.append(circle_element)

# -- Audio Feedbacks
# Define Sounds
pos_feedback = stimuli.Audio(POSITIVE_FEEDBACK)
neg_feedback = stimuli.Audio(NEGATIVE_FEEDBACK)

# ************************************************************************************************************************************
# Preload all necessary stimuli
# ************************************************************************************************************************************

# Visuals
container_figure.preload()
container_figure_response.preload()
for k in range(len(circles)):
    circles[k].preload()
photodiode.preload()
blankscreen.preload()

# Sounds
pos_feedback.preload()
neg_feedback.preload()



# ************************************************************************************************************************************
# Build the experimental structure
# ************************************************************************************************************************************

# -- Instructions
instructions = stimuli.TextScreen(
    "Instructions",
    f"""Vous allez voir des séquences de 12 positions. 
    Mémorisez les et reproduisez les quand la croix de fixation devient bleue.""", 
    heading_size=size_text,position=origin_position, text_size=size_text)

for index, name in enumerate(experiment_seq_names):
    
    # -- Create block
    b= design.Block()
    b.set_factor("block",index)
    b.set_factor("sequenceName",name)
    
    # -- Fill the block with trials
    for i in range(nb_trials_per_sequence):
        t=design.Trial()
        t.set_factor('block',b.get_factor('block'))
        t.set_factor('sequenceName',b.get_factor('sequenceName'))
        t.set_factor('trial', i)
        t.set_factor('PresentedSequence',str(experiment_seq_expression[index][i]))
        
        # -- Fill the trial with stimuli shown
        for pos in experiment_seq_expression[index][i]:
            # circle_index=experiment_seq_expression[index][i][k]
            # t.add_stimulus(circles[circle_index])
            t.add_stimulus(circles[pos])
            
            
        b.add_trial(t)
    
    exp.add_block(b)
    


# ************************************************************************************************************************************
# Run the experiment
# ************************************************************************************************************************************

control.start()
instructions.present()
exp.keyboard.wait()

for block in exp.blocks:
    block_nb=block.get_factor('block')
    for trial in block.trials:
        # -- Trial timeline
        # -- Check if it's time for resting state
        trial_nb=trial.get_factor('trial')
        
        
        # 0 - Clear screen
        exp.screen.clear()
        exp.screen.update()
        
        # 1 - Present Figure for 500 ms 
        container_figure.present()
        trigger=Trigger_codes['fixation']
        
        if MEG:
            port2send.send(data=trigger)
        
        exp.clock.wait(500)
        
        if MEG:
            port2send.send(data=0)
        
        # 2 - Present sequence
        presented_sequence=experiment_seq_expression[block_nb][trial_nb]

        for k in range(sequence_length):
            start = time.time()*1000
            # - Hexagon on screen
            container_figure.present()
            # - FLash point: circle appears
            #photodiode.present(clear=False,update=False)
            trial.stimuli[k].present(clear=False)
            trigger_key=f"{trial.get_factor('sequenceName')}-{presented_sequence[k]+1}"
            trigger = Trigger_codes[trigger_key]
            if MEG:
                trigger_key=f"{trial.get_factor('sequenceName')}-{presented_sequence[k]+1}"
                trigger = Trigger_codes[trigger_key]
                port2send.send(data=trigger)
            exp.clock.wait(SOS)
            
            if MEG:
                port2send.send(data=0)
            
            # - FLash point: circle disappears
            container_figure.present()
            true_SOS = time.time()*1000 - start
            exp.clock.wait(SOA-true_SOS)
            
            #print("[true_SOS] time.time()*1000-start : ",time.time()*1000-start)

        # Check for exit key during the break
        key_exit, rt_exit = exp.keyboard.wait_char(['q', 'ESCAPE'], duration=duration_break)
        if key_exit in ['q', 'ESCAPE']:
            control.end('Ending Experiment ...')
            break
        
        
        # 4 - Response Phase Starts
        # Holder objects for Behavioral Responses
        response_times=[]
        pressed_positions=[]
        # Variable to track time and score
        score=0
        position, rt = 99999,99999
        response_phase=True
        
        # Show Hexagon with blue cross
        container_figure_response.present()
        trigger=Trigger_codes['fixation_blue']
        
        if MEG:
            port2send.send(data=trigger)
            exp.clock.wait(duration_trigger)
            port2send.send(data=0)
        
        
        # Listen For responses
        while len(response_times)<sequence_length and response_phase:
            container_figure_response.present()
            if not MEG:
                key, rt = exp.keyboard.wait_char(list_btn, duration=duration_maximumAllowed)
            else:
                try:
                    port_code, rt = response_MEG.wait(duration=duration_maximumAllowed)

                    # We want to prevent accidental double clicking
                    next_click = time.time()*1000
                
                    if rt is not None and rt <=150:
                        print('too quick: double tap')
                        continue
                
                    key = response_funcs.trigger_to_forp[port_code]
                    print('Key = ', key)
                    last_click=next_click
                    #print('port_code',port_code)
                    #print('response_MEG.wait(duration=duration_maximumAllowed)',response_MEG.wait(duration=duration_maximumAllowed))
                    #print('response_funcs.trigger_to_forp[port_code] = ',response_funcs.trigger_to_forp[port_code])
                    
                except KeyError:
                    # Handle the case where participant did not respond
                    rt=duration_maximumAllowed
                    key=None
                    feedback = stimuli.TextLine("Aucune réponse", position=origin_position, text_size=size_text)
                    feedback.present()
                    neg_feedback.play()
                    exp.clock.wait(500)
                    response_phase = False
                    print('no response registered')

            if key is not None and key in response_mappings.keys():
                #print('position=response_mappings[key]         ',response_mappings[key])
                position=response_mappings[key]
                #photodiode.present(clear=False,update=False)
                circles[response_mappings[key]].present(clear=False)
                exp.clock.wait(100)
                container_figure_response.present()
                #FIXME MEG : activate
                print(f"BUTTON NUMBER {position} PRESSED\n")
            else:
                print('None key detected or wrong key : ',key)
                position=-1
                


            response_times.append(rt)
            pressed_positions.append(position)
        
        
        
        # 5 - Compute score and provide feedback
        if response_phase:
            
            distance=jellyfish.damerau_levenshtein_distance(u''.join(str(n) for n in pressed_positions),
                                                          u''.join(str(n) for n in presented_sequence))
            score = (6 - distance) / 6  # value between -1 and 1
            won = 100 * score
            
            if MEG:
                if distance==0:
                    trigger=Trigger_codes['win']
                else:
                    trigger=Trigger_codes['loss']
                port2send.send(data=trigger)
            exp.clock.wait(SOS)
            if MEG:
                port2send.send(data=0)
            
            
            if won <= 0:
                neg_feedback.play()
                feedback = stimuli.TextLine("Essai manqué !", position=origin_position, text_size=size_text)
                feedback.present()
            else:
                pos_feedback.play()
                feedback = stimuli.TextLine("Vous avez gagné %i points !" % (int(won)), position=origin_position, text_size=size_text)
                feedback.present()
        
        exp.clock.wait(duration_feedback)

        blankscreen.present()
        exp.clock.wait(ITI)
        
        exp.data.add([
            block.get_factor('block'),
            block.get_factor('sequenceName'),
            trial.get_factor('trial'),
            trial.get_factor('PresentedSequence'),
            pressed_positions,
            response_times,
            score
        ])
        
        # DeBUG
        # print('trial_nb : ',trial_nb)
        # print('nb_trials_per_sequence : ',nb_trials_per_sequence)
        # print('trial_nb == nb_trials_per_sequence ', trial_nb ==nb_trials_per_sequence)
        print('---------------------------------------')
        if trial_nb ==(nb_trials_per_sequence-1): # FIXME for MEG check the correction
            print('We are in the resting state break')
            stimuli.TextScreen('Phase Repos',
                """Vous pouvez fermer les yeux.
                    Nous vous préviendrons quand l experience reprendra.""", 
                    heading_size=size_text,position=origin_position, text_size=size_text).present()
            
            exp.clock.wait(duration_padding)
            blankscreen.present()
            exp.clock.wait(duration_padding)
            key_exit, rt_exit = exp.keyboard.wait_char(['q'], duration=duration_resting)
            if key_exit:
                control.end('Ending experiment')
                break
            # AFTER THE RESTING STATE, WE START A NEW BLOCK
            stimuli.TextLine('Nous allons reprendre l experience', position=origin_position, text_size=size_text).present()
            key_next_block, rt_next_block = exp.keyboard.wait_char(['n'])
            if key_next_block:
                stimuli.TextLine('Reprise de l experience', position=origin_position, text_size=size_text).present()
                exp.clock.wait(duration_padding)
                container_figure.present()
                exp.clock.wait(duration_pre_start)
                
exp.keyboard.process_control_keys()
exp.clock.wait(500)
print('END OF EXPERIMENT -- Congratulations. Remember to let out your participant !')
control.end()



"""
Change logs

-- Version: 1.1.3 -- 08.07.2024
- Changed the value of the duration for the break before presentation after resting state. duration_pre_start went from 2000 to 1000
- Deleted the extra 6 seconds wait in between presentation and reproduction
- On enlève la phase de disparition des place holder avant début de la présentation

-- Version: 1.1.2 -- 04.07.2024
- Changed the value of the duration for the break before presentation after resting state. duration_pre_start went from 5000 to 2000


-- Version: 1.1 -- 27.06.2024
- Added the function that sends the triggers

"""