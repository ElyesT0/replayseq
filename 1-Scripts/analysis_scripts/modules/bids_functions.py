
from modules.params import *
from fpdf import FPDF


def extract_events_and_event_IDs_neurospin(raw,event_dict=event_dict):
    events_presentation=mne.find_events(raw,mask_type = "not_and",mask = 2**6+2**7+2**8+2**9+2**10+2**11+2**12+2**13+2**14+2**15, verbose=False, min_duration=0.1)
    
    return events_presentation, event_dict 
    
    

def extract_events_and_event_IDs_ICM(raw):
    """
    The trigger codes are
    9 for fixation,
    trigger_seq + item + 1 for the items of the sequence,
    99 at the beginning of the reproduction phase and
    128 at the beginning of the resting state phase.
    """

    events = mne.find_events(raw, min_duration=0.01)
    # on élimine les évènements de feedback de score (10*DL distance)
    events = events[events[:,2]%10!=0,:]
    events_fixation = events[events[:,2]==9,:]
    events_sequence_presentation = events[[events[i,2]>10 and events[i,2]<97 for i in range(len(events))]]
    events_fixation_blue = events[events[:,2]==99]
    events_resting_phase = events[events[:,2]==128]
    events_of_interest = np.vstack([events_fixation,events_fixation_blue,events_sequence_presentation,events_resting_phase])

    dict_fixation = {'Fixation':9}
    dict_reproduction = {'Reproduction':99}
    seqID = int(np.floor(events_sequence_presentation[0,2]/10)*10)
    sequence_events = {10: 'Rep2',20: 'CRep2',30: 'Rep3',40: 'CRep3',50: 'Rep4',60: 'CRep4',70: 'RepEmbed',80: 'C1RepEmbed',90: 'C2RepEmbed'}
    dict_sequences = {'SequenceID-%s/Position-%i'%(sequence_events[seqID],i):seqID+i  for i in range(1,7)}
    dict_resting_state = {'Resting_state':128}
    event_ids_dict =  dict_fixation | dict_reproduction | dict_sequences | dict_resting_state

    return events_of_interest, event_ids_dict



def prepare_data_for_mne_bids_pipeline(sub,path_exp = "/Volumes/T5_EVO/1-experiments",local=True):
    # local (bool): If True, works for my local machine. If false, works on remote machine that pulled the script from github.
    
    if local:
        path_root=os.path.join(path_exp,"REPLAYSEQ")
        path_json_file = os.path.join("/Users/et/Documents/UNICOG/2-Experiments/replayseq/1-Scripts/analysis_scripts/modules/objects/bad_channels.json")
    else:
        path_root=os.path.join(path_exp,"replayseq")
        path_json_file = os.path.join(path_root,"1-Scripts/analysis_scripts/modules/objects/bad_channels.json")
        
    path_data_raw=os.path.join(path_root,"2-Data/raw")
    path_BIDS=os.path.join(path_root,"2-Data/BIDS")
    
    
    # Open JSON bad_channels object
    with open(path_json_file, 'r') as file:
        bad_channels_dict = json.load(file)

    subject=f'sub-{sub:02}'
    
    lab=bad_channels_dict[f'{subject}']['lab']
    
    # Define Path    
    if lab=='icm':
        original_data_path=os.path.join(path_data_raw,'Data_ICM/')
    #path_root = "/Volumes/T5_EVO/1-experiments/REPLAYSEQ/2-Data/raw"
        
    else:
        original_data_path=os.path.join(path_data_raw,'Data_neurospin/')


    for run in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']:
        
        print("--- saving in bids format run %s ------")
        if lab=='icm':
            data_path = original_data_path+subject+ '/run%s.fif'%run
        else:
            data_path=original_data_path+subject+f'/run{run}_raw.fif'
        raw = mne.io.read_raw_fif(data_path,allow_maxshield=True,preload=True)
        n_chans=raw.get_data().shape[0]
        raw.pick(np.where(['EEG' not in raw.info['ch_names'][i] for i in range(n_chans)])[0])
        
        if lab=='icm':
            raw.set_channel_types({'BIO001': 'eog'})
            raw.set_channel_types({'BIO002': 'eog'})
            raw.set_channel_types({'BIO003': 'ecg'})
            events, event_ids = extract_events_and_event_IDs_ICM(raw)
            
        else:
            events, event_ids = extract_events_and_event_IDs_neurospin(raw)
        bids_path = BIDSPath(subject=f'{sub:02}', task='reproduction', run=run, datatype='meg', root=path_BIDS)

        # Append bad channels from JSON object
        raw.info['bads']=bad_channels_dict[subject]['run'+run]
        
        write_raw_bids(raw, bids_path=bids_path,allow_preload=True,format='FIF',events=events,event_id=event_ids,overwrite=True)

        # write MEG calibration files
        if lab=='icm':
            ct_fname = path_BIDS + "/calibration_files/calibration_icm/ct_sparse.fif"
            cal_fname = path_BIDS + "/calibration_files/calibration_icm/sss_cal_3101_160108.dat"
        else:
            ct_fname = path_BIDS + "/calibration_files/calibration_neurospin/ct_sparse.fif"
            cal_fname = path_BIDS + "/calibration_files/calibration_neurospin/sss_cal_3176_20240123_2.dat"


        write_meg_calibration(calibration=cal_fname,bids_path=bids_path)
        write_meg_crosstalk(fname=ct_fname,bids_path=bids_path)

def inspect_raw(sub_nb, run, path_root=root_path,verbose=False,bad_channels_test=[]):
    # bad_channels_test : enter group of usually recognized bad channels. Makes it easier to check if they are still bad accross runs.
    
    # Open JSON bad_channels object
    with open(path_json_file, 'r') as file:
        bad_channels_dict = json.load(file)
    
    lab=bad_channels_dict[f'sub-{sub_nb:02}']['lab']
    
    if lab=='icm':
        path_raw=os.path.join(path_root,'Data_ICM')
        path_raw_file=os.path.join(path_raw,f'sub-{sub_nb:02}/run{run:02}.fif')
        
    else:
        path_raw=os.path.join(path_root,'Data_neurospin')
        path_raw_file=os.path.join(path_raw,f'sub-{sub_nb:02}/run{run:02}_raw.fif')
        
        
    
    # Open the raw object
    raw=mne.io.read_raw_fif(path_raw_file, allow_maxshield=True, preload=True,verbose=verbose)
    raw.info['bads']=bad_channels_test
    raw_filter = raw.copy().notch_filter(freqs=[50,100,150])
    
    # 1 - Plot the raw object for the given subjet / run.
    raw_filter.plot(n_channels=30)
    
    # 2 - Plot the PSD to note outliers 
    #raw.pick_types(eeg=False)
    raw_filter.compute_psd().plot()
    
    
    # eventuellement prendre raw.crop() 100 secondes au milieu
    
def prepare_json_bad_channels(path_json_file, sub):
    # Load the JSON file
    with open(path_json_file, 'r') as file:
        bad_channels_dict = json.load(file)
    
    # Create the subject key if it doesn't exist
    for k in range(1, sub + 1):
        subject_key = f'sub-{k:02}'
        if subject_key not in bad_channels_dict:
            bad_channels_dict[subject_key] = {}
    
    # Update each subject with 18 runs
    for subject_key in bad_channels_dict:
        if 'lab' in bad_channels_dict[subject_key]:
            lab_value = bad_channels_dict[subject_key]['lab']
            
        else:
            lab_value = 'neurospin'  # Provide a default lab value if needed
        
        # Update runs without overwriting existing data
        for i in range(1, 19):
            run_key = f'run{str(i).zfill(2)}'
            if run_key not in bad_channels_dict[subject_key]:
                bad_channels_dict[subject_key][run_key] = []
        
        # Ensure 'lab' value is preserved or set to default
        bad_channels_dict[subject_key]['lab'] = lab_value

    # Save the modified dictionary back to the JSON file
    with open(path_json_file, 'w') as file:
        json.dump(bad_channels_dict, file, indent=4)
        


# Function to generate the BIDS report
def generate_bids_report(subject_id, run, raw, annot_muscle, bids_path, output_path):
    # Ensure participant's directory exists
    participant_path = os.path.join(output_path, f"sub-{subject_id:02}")
    if not os.path.exists(participant_path):
        os.makedirs(participant_path)

    # Create a PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Header
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt=f"BIDS Processing Report: sub-{subject_id:02} run-{run:02}", ln=True, align='C')
    pdf.ln(10)

    # File Information
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Raw file: {path_raw}/run{run}_raw.fif", ln=True)
    pdf.cell(200, 10, txt=f"BIDS path: {path_BIDS}", ln=True)
    pdf.ln(5)

    # Annotations
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(200, 10, txt="Annotations:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Total annotations: {len(raw.annotations)}", ln=True)
    pdf.cell(200, 10, txt=f"Types: {set(annot['description'] for annot in raw.annotations)}", ln=True)
    pdf.cell(200, 10, txt=f"Total duration: {sum(annot['duration'] for annot in raw.annotations)} seconds", ln=True)

    # Plot raw data with annotations
    fig = raw.plot(show=False)
    fig.savefig(f"{participant_path}/sub-{subject_id:02}_run-{run:02}_annotations.png")
    pdf.image(f"{participant_path}/sub-{subject_id:02}_run-{run:02}_annotations.png", x=10, y=None, w=180)
    pdf.ln(5)

    # Save the PDF report
    report_file = os.path.join(participant_path, f"sub-{subject_id:02}_run-{run:02}_BIDS_report.pdf")
    pdf.output(report_file)
    print(f"Report saved to {report_file}")