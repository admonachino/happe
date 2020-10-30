%% DETERMINE PATH TO DATA
src_folder_name = input('Path to the folder containing the datasets:\n> ','s') ;
cd (src_folder_name);
%% DETERMINE IF USER HAS PRESETS
disp('Load pre-existing set of input parameters? [Y/N]') ;
while true
    % request and store user input
    pre_exist = input('> ', 's') ;
    % if yes, set indicating variable to 1
    if strcmpi(pre_exist, "y")
        pre_exist = 1 ;
        break ;
    % if no, set indicating variable to 0
    elseif strcmpi(pre_exist, "n")
        pre_exist = 0 ;
        break ;
    % otherwise, input is invalid - prompt user to enter valid input
    else
        disp("Invalid input: please enter Y or N.") ;
    end
end
%% IF THE USER HAS PRESETS...
if pre_exist
    % request, collect, and add path to the input parameters
    preparam_folder_name = input('Path to the folder containing the input parameters:\n> ','s') ;
    addpath(preparam_folder_name) ;
    % collect the name of the file containing parameters, and load it
    load(input('Name of file containing pre-existing set of parameters:\n> ', 's')) ;
    % DETERMINE IF USER HAS ANY INPUTS THEY WOULD LIKE TO CHANGE
    while true
        user_input = input('Change existing parameters? [Y/N]\n> ', 's') ;
        if strcmpi(user_input, 'y')
            while true
                u_input = input(['Parameter to change:\n' ...
                    'aquisition layout, rest/task data, channels of interest, data file format,\n' ...
                    'line noise frequency, downsampling, visualization, segmentation, interpolation\n' ...
                    'segment rejection, rereferencing, save format\n' ...
                    'Enter "done" (without quotations) when finished changing parameters.\n' ...
                    '> '], 's') ;
                if strcmpi(u_input, 'aquisition layout')
                    layout_type = determ_aquiLayout() ;
                elseif strcmpi(u_input, 'rest/task data')
                    [task_EEG_processing, ERP_analysis] = determ_rVt() ;
                    if ERP_analysis
                        [ERP_lowpass_cutoff, ERP_highpass_cutoff] = determ_erpCutoff() ;
                    end
                elseif strcmpi(u_input, 'channels of interest')
                    chan_IDs = determ_chanIDs(layout_type) ;
                elseif strcmpi(u_input, 'data file format')
                    netstationRAWformat = determ_fileFormat() ;
                    if ~netstationRAWformat
                        [potential_eeg_var_names, sampling_rate_varname] = fileFormat_matlabArray() ;
                        if task_EEG_processing == 1
                        task_event_info_location = fileFormat_matlabArray_txtLocation() ;
                        end
                    elseif netstationRAWformat
                        task_onset_tags = fileFormat_dotRAW() ;
                    end
                elseif strcmpi(u_input, 'line noise frequency')
                    line_noise = input(['Frequency of electrical (line) noise in Hz:\n' ...
                        'USA data probably = 60; Otherwise, probably = 50\n' ...
                        '> ']) ;
                elseif strcmpi(u_input, 'downsampling')
                    downsample_data = determ_downsample() ;
                    if downsample_data
                        downsample_freq = input(['Frequency for resampling the data in Hz:\n' ...
                        '> ']) ;
                    end
                elseif strcmpi(u_input, 'visualization')
                    pipeline_visualizations_semiautomated = determ_visualizations() ;
                    if pipeline_visualizations_semiautomated
                        [vis_freq_min, vis_freq_max, freq_to_plot] = visulization_param() ;
                        if ERP_analysis
                            [vis_time_start, vis_time_end] = visualization_param_erp() ;
                        end
                    end
                elseif strcmpi(u_input, 'segmentation')
                    segment_data = determ_segment() ;
                    if segment_data
                        if task_EEG_processing
                            [task_segment_start, task_segment_end] = segment_task() ;
                            if ERP_analysis
                                [task_offset, baseline_correction] = segment_erp() ;
                                if baseline_correction
                                    [baseline_corr_start, baseline_corr_end] = baseline_corr() ;
                                end
                            end
                        elseif ~task_EEG_processing
                            segment_length = segment_rest() ;
                        end
                    end
                elseif strcmpi(u_input, 'interpolation')
                    segment_interpolation = determ_interpolation() ;
                elseif strcmpi(u_input, 'segment rejection')
                    segment_rejection = determ_segReject() ;
                    if segment_rejection
                        [reject_min_amp, reject_max_amp] = segReject_amp() ;
                        ROI_channels_only = segReject_ROI() ;
                        if ROI_channels_only
                            ROI_channels = segReject_ROIchannels() ;
                        end
                        if task_EEG_processing && ~netstationRAWformat
                        user_selected_trials = segReject_trials() ;
                        end
                    end
                elseif strcmpi(u_input, 'rereferencing')
                    average_rereference = determ_reref() ;
                    if ~average_rereference
                        NO_AVERAGE_REREF_channel_subset = reref_nonaverage() ;
                    end
                elseif strcmpi(u_input, 'save format')
                    save_as_format = determ_saveFormat() ;
                elseif strcmpi(u_input, 'done')
                    break ;
                else
                    disp("Invalid input: please enter a valid parameter.") ;
                end
            end
        elseif strcmpi(user_input, 'n')
           break ;
       else
           disp("Invalid input: please enter Y or N.") ;
       end
    end
    
%% IF THE USER DOES NOT HAVE PRESETS...
elseif ~pre_exist
    % DETERMINE AQUISITION LAYOUT
    layout_type = determ_aquiLayout() ;

    % DETERMINE IF RESTING STATE OR TASK DATA
    [task_EEG_processing, ERP_analysis] = determ_rVt() ;
    if ERP_analysis
        [ERP_lowpass_cutoff, ERP_highpass_cutoff] = determ_erpCutoff() ;
    end

    % COMPILE CHANNELS OF INTEREST
    chan_IDs = determ_chanIDs(layout_type) ;

    % DETERMINE IF NETSTATION .RAW OR MATLAB ARRAY
    netstationRAWformat = determ_fileFormat() ;
    if ~netstationRAWformat
        [potential_eeg_var_names, sampling_rate_varname] = fileFormat_matlabArray() ;
        if task_EEG_processing == 1
            task_event_info_location = fileFormat_matlabArray_txtLocation() ;
        end
    elseif netstationRAWformat
        task_onset_tags = fileFormat_dotRAW() ;
    end

    % DETERMINE LINE NOISE FREQUENCY
    line_noise = input(['Frequency of electrical (line) noise in Hz:\n' ...
        'USA data probably = 60; Otherwise, probably = 50\n' ...
        '> ']) ;
    
    % DETERMINE IF DOWNSAMPLING
    downsample_data = determ_downsample() ;
    % determine frequency for resampling
    if downsample_data
        downsample_freq = input(['Frequency for resampling the data in Hz:\n' ...
            '> ']) ;
    end
    
    % DETERMINE IF RUNNING WITH VISUALIZATIONS
    pipeline_visualizations_semiautomated = determ_visualizations() ;
    if pipeline_visualizations_semiautomated
        [vis_freq_min, vis_freq_max, freq_to_plot] = visulization_param() ;
        if ERP_analysis
            [vis_time_start, vis_time_end] = visualization_param_erp() ;
        end
    end
    
    % DETERMINE IF SEGMENTING
    segment_data = determ_segment() ;
    if segment_data
        if task_EEG_processing
            [task_segment_start, task_segment_end] = segment_task() ;
            if ERP_analysis
                [task_offset, baseline_correction] = segment_erp() ;
                if baseline_correction
                    [baseline_corr_start, baseline_corr_end] = baseline_corr() ;
                end
            end
        elseif ~task_EEG_processing
            segment_length = segment_rest() ;
        end
    end

    % DETERMINE IF INTERPOLATING
    segment_interpolation = determ_interpolation() ;

    % DETERMINE IF REJECTING SEGMENTS
    segment_rejection = determ_segReject() ;
    if segment_rejection
        [reject_min_amp, reject_max_amp] = segReject_amp() ;
        ROI_channels_only = segReject_ROI() ;
        if ROI_channels_only
            ROI_channels = segReject_ROIchannels() ;
        end
        if task_EEG_processing && ~netstationRAWformat
            user_selected_trials = segReject_trials() ;
        end
    end
    
    % DETERMINE TYPE OF RE-REFERENCING
    average_rereference = determ_reref() ;
    if ~average_rereference
        NO_AVERAGE_REREF_channel_subset = reref_nonaverage() ;
    end
                
    % DETERMINE SAVE FORMAT FOR PROCESSED DATA
    save_as_format = determ_saveFormat() ;
end

%% Save input variables
%make folder for intermediate unfiltered files
if ~isdir ([src_folder_name filesep 'input_parameters'])
	mkdir ([src_folder_name filesep 'input_parameters']);
end
cd input_parameters ;
input_vars = {'src_folder_name','layout_type','chan_IDs','pipeline_visualizations_semiautomated' ...
	'vis_freq_min', 'vis_freq_max', 'freq_to_plot', 'vis_time_start', 'vis_time_end' ...
	'task_EEG_processing', 'ERP_analysis', 'ERP_highpass_cutoff', 'ERP_lowpass_cutoff' ...
	'sampling_rate_varname', 'line_noise', 'netstationRAWformat' ...
	'potential_eeg_var_names', 'task_event_info_location', 'task_onset_tags' ...
	'user_selected_trials', 'segment_data', 'task_segment_start', 'task_segment_end' ...
	'task_offset', 'baseline_correction', 'baseline_corr_start', 'baseline_corr_end' ...
	'segment_length', 'segment_interpolation', 'segment_rejection', 'reject_min_amp' ...
	'reject_max_amp', 'ROI_channels_only', 'ROI_channels', 'average_rereference' ...
    'NO_AVERAGE_REREF_channel_subset', 'save_as_format', 'downsample_data', 'downsample_freq'};
param_file = ['inputParameters_' datestr(now, 'dd-mm-yyyy') '.mat'] ;
for indx = 1:length(input_vars)
	if exist(input_vars{indx}, 'var')
        if exist(param_file)
            save(param_file, input_vars{indx}, '-append') ;
        else
        	save(param_file, input_vars{indx}) ;
        end
	end
end

%% MAKE OUTPUT FOLDERS
% make folder for intermediate unfiltered files
if ~isdir ([src_folder_name filesep 'intermediate0_unfiltered']) && ERP_analysis == 1
    mkdir ([src_folder_name filesep 'intermediate0_unfiltered']);
end

% make folder for intermediate waveleted files
if ~isdir ([src_folder_name filesep 'intermediate1_wavclean'])
    mkdir ([src_folder_name filesep 'intermediate1_wavclean']);
end

% make folder for post-ICA, uninterpolated files
if ~isdir ([src_folder_name filesep 'intermediate2_ICAclean'])
    mkdir ([src_folder_name filesep 'intermediate2_ICAclean']);
end

% make folder for segment-level files (if user selects segmentation option)
if ~isdir ([src_folder_name filesep 'intermediate3_segmented']) && segment_data ==1
    mkdir ([src_folder_name filesep 'intermediate3_segmented']);
end

% make folder for final preprocessed files
if ~isdir ([src_folder_name filesep 'processed'])
    mkdir ([src_folder_name filesep 'processed']);
end

%% ADD RELEVANT FOLDERS TO PATH
% add HAPPE script path
happe_directory_path = fileparts(which(mfilename('fullpath')));

% add EEGLAB path
% will eventually allow users to set own eeglab path --
% for now, assume using eeglab14_0_0b included in HAPPE
eeglab_path = [happe_directory_path filesep 'Packages' filesep 'eeglab14_0_0b'];

% add HAPPE subfolders path
addpath([happe_directory_path filesep 'acquisition_layout_information'],[happe_directory_path filesep 'scripts'],...
    eeglab_path,genpath([eeglab_path filesep 'functions']));
rmpath(genpath([eeglab_path filesep 'functions' filesep 'octavefunc']));

% add EEGLAB plugin folders to path
plugin_directories = dir([eeglab_path filesep 'plugins']);
plugin_directories = strcat(eeglab_path,filesep,'plugins',filesep,{plugin_directories.name},';');
addpath([plugin_directories{:}]);

% add cleanline path
if exist('cleanline','file')
    cleanline_path = which('eegplugin_cleanline.m');
    cleanline_path = cleanline_path(1:findstr(cleanline_path,'eegplugin_cleanline.m')-1);
    addpath(genpath(cleanline_path));
else
    error('Please make sure cleanline is on your path');
end

% set path to sensor layout
switch layout_type
    case 1
        chan_locations = [happe_directory_path filesep 'acquisition_layout_information' filesep 'GSN-HydroCel-32.sfp'];
    case 2
        chan_locations = [happe_directory_path filesep 'acquisition_layout_information' filesep 'GSN64v2_0.sfp'];
    case 3
        chan_locations = [happe_directory_path filesep 'acquisition_layout_information' filesep 'GSN-HydroCel-128.sfp'];
    otherwise
        error ('Invalid sensor layout selection. Users wishing to use an unsupported layout can run HAPPE through\n%s',...
            ' the Batch EEG Automated Processing Platform (BEAPP),  as described in the HAPPE manuscript.')
end

%% COLLECT AND PREPARE TO RUN DATA
% Go to the folder with data
cd (src_folder_name);
% establish file extension
if netstationRAWformat == 1
    src_file_ext = '.raw';
else
    src_file_ext = '.mat';
end

% pull the file names to feed script
FileNames=dir(['*' src_file_ext]);
FileNames={FileNames.name};

% if event EEG processing with separate stimulus txt file...
if netstationRAWformat == 0 && task_EEG_processing == 1
    % locate the file and pull the stim names
    cd (task_event_info_location);
    stim_file_ext = '.txt';
    StimNames=dir(['*' stim_file_ext]);
    StimNames={StimNames.name};
end

% intialize data quality report metrics
chan_index=[1:length(chan_IDs)];
Number_ICs_Rejected=[];
Number_Good_Channels_Selected=[];
Interpolated_Channel_IDs=[];
Percent_ICs_Rejected=[];
Percent_Variance_Kept_of_Post_Waveleted_Data=[];
File_Length_In_Secs=[];
Number_Channels_User_Selected=[];
Percent_Good_Channels_Selected=[];
Median_Artifact_Probability_of_Kept_ICs=[];
Mean_Artifact_Probability_of_Kept_ICs=[];
Range_Artifact_Probability_of_Kept_ICs=[];
Min_Artifact_Probability_of_Kept_ICs=[];
Max_Artifact_Probability_of_Kept_ICs=[];
Number_Segments_Post_Segment_Rejection=[];
Channels_Interpolated_For_Each_Segment = {};

%% iterate the following preprocessing pipeline over all your data files:
for current_file = 1:length(FileNames)
    cd (src_folder_name);
    
    %% LOAD FILE AND GET SAMPLING RATE
    % import data into eeglab and store the file's length in seconds for outputting later
    % read in task files in .raw format
    if task_EEG_processing == 1 && netstationRAWformat == 1
        EEGloaded = pop_readegi(FileNames{current_file}, [],[],'auto');
        events=EEGloaded.event;
        orig_event_info=EEGloaded.urevent;
        srate=double(EEGloaded.srate);
        
    % read in task or baseline files in matlab format
    elseif  netstationRAWformat == 0
        load(FileNames{current_file});
        srate=double(eval(sampling_rate_varname{1}));
        file_eeg_vname = intersect(who,potential_eeg_var_names);
        try
            EEGloaded = pop_importegimat(FileNames{current_file}, srate, 0.00, file_eeg_vname{1});
        catch err_msg
            if strcmp(err_msg.identifier,'MATLAB:badsubscript')
                error('sorry, could not read the variable name of the EEG data, please check your file')
            else
                error(err_msg.message);
            end
        end
        
        % load events for .mat file
        if task_EEG_processing == 1
           cd (task_event_info_location);
                EEGloaded = pop_importevent( EEGloaded, 'append','no','event',StimNames{current_file},'fields',...
                {'type' 'latency' 'status'},'skipline',1,'timeunit',1E-3,'align',NaN);
            orig_event_info=EEGloaded.urevent;
            events=EEGloaded.event;
        end
        
    else % throw error for baseline .raw or for unsupported file formats
        error ('file format type unsupported or running non-task with raw file');
    end
    
    EEGloaded.setname='rawEEG';
    EEG = eeg_checkset(EEGloaded);
    
    File_Length_In_Secs(current_file)=EEG.xmax;
    
    % edit channel locations (does not import properly from netstation by default)
    EEG = pop_chanedit(EEG, 'load',{chan_locations 'filetype' 'autodetect'});
    % EEG = eeg_checkset(EEG);
    
    %load 10-20 EEG system labels for electrode names (for MARA to reference)
    load('happe_netdata_lib.mat')
    
    %
    switch layout_type
        %for 32 channel nets
        case 1
            for i=1:length(netdata_lib.net32.lead_nums_sub)
                EEG=pop_chanedit(EEG, 'changefield',{netdata_lib.net32.lead_nums_sub(i)  'labels' netdata_lib.net32.lead_list_sub{i}});
            end
            
            %for 64 channel nets
        case 2
            for i=1:length(netdata_lib.net64.lead_nums_sub)
                EEG=pop_chanedit(EEG, 'changefield',{netdata_lib.net64.lead_nums_sub(i)  'labels' netdata_lib.net64.lead_list_sub{i}});
            end
            
            % for 128 channel nets
        case 3
            for i=1:length(netdata_lib.net128.lead_nums_sub)
                EEG=pop_chanedit(EEG, 'changefield',{netdata_lib.net128.lead_nums_sub(i)  'labels' netdata_lib.net128.lead_list_sub{i}});
            end
        otherwise
            error('Sensor layout not currently supported. HAPPE can be run using additional sensor layouts in BEAPP (see description in HAPPE header)');
    end
    
    % EEG = eeg_checkset(EEG);
    
    if ERP_analysis == 1
        %save EEG out here for ERP analyses, HAPPE will act on this unfiltered data later to undo the filtering step below (required for ICA)
        EEG_unfiltered = pop_saveset(EEG, 'filename',strrep(FileNames{current_file}, src_file_ext,'_unfiltered.set'),'filepath',[src_folder_name filesep 'intermediate0_unfiltered']);
    end
    
    %% filter the data with 1hz highpass (for srate 250), bandpass 1hz-249hz (for srate 500, ICA doesn't reliably work well with frequencies above 250hz)
    if srate<500
        EEG = pop_eegfiltnew(EEG, [],1,[],1,[],0);
    elseif srate >= 500
        EEG = pop_eegfiltnew(EEG, 1,250,[],0,[],0); % ADM: change filter to 150 ?
    end
    
    EEG.setname='rawEEG_f';
    % EEG = eeg_checkset(EEG);
    
    %% select EEG channels of interest for analyses and 10-20 channels from the list you specified at the top of the script
    EEG = pop_select(EEG,'channel', chan_IDs);
    EEG.setname='rawEEG_f_cs';
    
    % EEG = eeg_checkset(EEG);
    full_selected_channels = EEG.chanlocs;
    
    %% reduce line noise in the data (note: may not completely eliminate, re-referencing helps at the end as well)
    EEG = pop_cleanline(EEG, 'bandwidth',2,'chanlist',chan_index,'computepower',1,'linefreqs',...
        [line_noise, line_noise*2] ,'normSpectrum',0,'p',0.01,'pad',2,'plotfigures',0,'scanforlines',1,'sigtype',...
        'Channels','tau',100,'verb',0,'winsize',4,'winstep',1, 'ComputeSpectralPower','False');
    EEG.setname='rawEEG_f_cs_ln';
    % EEG = eeg_checkset(EEG);
    
    % close window if visualizations are turned off
    if pipeline_visualizations_semiautomated == 0
        close gcf;
    end
    
    %% resample data to specified frequency
    if downsample_data == 1
        EEG = pop_resample(EEG, downsample_freq);
    end
    
    %% crude bad channel detection using spectrum criteria and 3SDeviations as channel outlier threshold, done twice
    EEG = pop_rejchan(EEG, 'elec',chan_index,'threshold',[-3 3],'norm','on','measure','spec','freqrange',[1 125]);
    EEG.setname='rawEEG_f_cs_ln_badc';
    % EEG = eeg_checkset(EEG);
    
    EEG = pop_rejchan(EEG, 'elec',[1:EEG.nbchan],'threshold',[-3 3],'norm','on','measure','spec','freqrange',[1 125]);
    EEG.setname='rawEEG_f_cs_ln_badc2x';
    % EEG = eeg_checkset(EEG);
    selected_good_channel_locations=EEG.chanlocs;
    
    %save the names of the rejected channels for output table after the pipeline finishes
    selected_channel_labels={selected_good_channel_locations.labels};
    bad_channels_removed= setdiff(chan_IDs, selected_channel_labels,'stable');
    if exist('ROI_channels', 'var')
        [~,ROI_indices_in_selected_chanlocs] = intersect(selected_channel_labels,ROI_channels,'stable');
    end
 
    %% run wavelet-ICA (ICA first for clustering the data, then wavelet thresholding on the ICs)
    % uses a soft, global threshold for the wavelets, wavelet family is coiflet (level 5), 
    % threshold multiplier 1 to remove more high frequency noise or 2 or ERP analyses
    % for details, see wICA.m function
    % higher threshold is more liberal 
    if ERP_analysis == 1
        threshmultiplier = 3;
    elseif ERP_analysis == 0
        threshmultiplier = 1;
    end
    
    try
        if pipeline_visualizations_semiautomated == 0
            [wIC, A, W, IC] = wICA(EEG,'runica', threshmultiplier, 0, [], 5, 'coif5'); % ADM: changed 5 to 6
        elseif pipeline_visualizations_semiautomated == 1
            [wIC, A, W, IC] = wICA(EEG,'runica', threshmultiplier, 1, srate, 5, 'coif5'); % ADM: changed 5 to 6
        end
    catch wica_err
        if strcmp ('Output argument "wIC" (and maybe others) not assigned during call to "wICA".',wica_err.message)
            error('Error during wICA, most likely due to memory settings. Please confirm your EEGLAB memory settings are set according to the description in the HAPPE ReadMe')
        else
           rethrow(wica_err)
        end
    end
    
    %reconstruct artifact signal as channelsxsamples format from the wavelet coefficients
    artifacts = A*wIC ;
    
    %reshape EEG signal from EEGlab format to channelsxsamples format if
    % original data was pre-segmented/3D
    EEG2D=reshape(EEG.data, size(EEG.data,1), []) ;
    
    %subtract out wavelet artifact signal from EEG signal
    wavcleanEEG=EEG2D-artifacts ;
    
    %save wavelet cleaned EEG data file to folder with extension _wavclean.mat
    cd ([src_folder_name filesep 'intermediate1_wavclean']);
    
    save(strrep(FileNames{current_file}, src_file_ext,'_wavclean.mat'),'wavcleanEEG')
    save(strrep(FileNames{current_file}, src_file_ext,'_prewav.mat'),'EEG2D')
    
    % save wavecleaned data into EEGLAB structure
    EEG.data = wavcleanEEG;
    EEG.setname='wavcleanedEEG';
    % EEG = eeg_checkset(EEG);
    
    %% run ICA to evaluate components this time
    EEG = pop_runica(EEG, 'extended',1,'interupt','on');
    % EEG = eeg_checkset(EEG);
    
    ICAweightstotransfer = EEG.icaweights;
    ICAspheretotransfer = EEG.icasphere;
    
    %save the ICA decomposition intermediate file before cleaning with MARA
    EEG = pop_saveset(EEG, 'filename',strrep(FileNames{current_file}, src_file_ext,...
        '_ICA.set'),'filepath',[src_folder_name filesep 'intermediate2_ICAclean']);
    
    %% use MARA to flag artifactual IComponents automatically if artifact probability > .5
    [~,EEG,~] = processMARA (EEG,EEG,EEG, [0, 0, pipeline_visualizations_semiautomated,...
        pipeline_visualizations_semiautomated , pipeline_visualizations_semiautomated] );
    
    EEG.reject.gcompreject = zeros(size(EEG.reject.gcompreject));
    %EEG.reject.gcompreject(EEG.reject.MARAinfo.posterior_artefactprob >= .5) = 1;
    ICs_with_reject_flagged = EEG.reject.gcompreject;
    EEG.setname='wavcleanedEEG_ICA_MARA';
    % EEG = eeg_checkset(EEG);
    
    % store MARA related variables to assess ICA/data quality    
    index_ICs_kept = find(EEG.reject.gcompreject == 0);
    median_artif_prob_good_ICs = median(EEG.reject.MARAinfo.posterior_artefactprob(index_ICs_kept));
    mean_artif_prob_good_ICs = mean(EEG.reject.MARAinfo.posterior_artefactprob(index_ICs_kept));
    range_artif_prob_good_ICs = range(EEG.reject.MARAinfo.posterior_artefactprob(index_ICs_kept));
    min_artif_prob_good_ICs = min(EEG.reject.MARAinfo.posterior_artefactprob(index_ICs_kept));
    max_artif_prob_good_ICs = max(EEG.reject.MARAinfo.posterior_artefactprob(index_ICs_kept));
    
    %store IC variables and calculate variance of data that will be kept after IC rejection:
    ICA_act = EEG.icaact;
    ICA_winv = EEG.icawinv;
    
    %variance of wavelet-cleaned data to be kept = varianceWav:
    [projWav, varianceWav] =compvar(EEG.data, ICA_act, ICA_winv,  index_ICs_kept);
    
    %% reject the ICs that MARA flagged as artifact
    artifact_ICs=find(EEG.reject.gcompreject == 1);
    EEG = pop_subcomp( EEG, artifact_ICs, 0);
    EEG.setname='wavcleanedEEG_ICA_MARA_rej';
    % EEG = eeg_checkset(EEG);
    
    %% save the post-MARA cleaned intermediate file before interpolating anything
    EEG = pop_saveset(EEG, 'filename',strrep(FileNames{current_file}, src_file_ext,...
        '_ICAcleanedwithMARA.set'),'filepath',[src_folder_name filesep 'intermediate2_ICAclean']);
    
    %% ERP analysis section ONLY:
    if ERP_analysis == 1
        
        %here we will effectively un-apply the 1 hz filter used for artifact
        % rejection to proceed with processing for ERP analyses
        %load the unfiltered EEG data saved from above
        EEG = EEG_unfiltered;
        EEG.setname='rawEEG_forERP';
        % EEG = eeg_checkset(EEG);
        
        %select EEG channels of interest for analyses and 10-20 channels
        EEG = pop_select(EEG,'channel', selected_channel_labels);
        EEG.setname='rawEEG_forERP_cs';
        % EEG = eeg_checkset(EEG);
        
        %reshape EEG signal from EEGlab format to channelsxsamples format if
        % original data was pre-segmented/3D
        EEG2DforERP=reshape(EEG.data, size(EEG.data,1), []);
       
        %subtract out wavelet artifact signal from the first processing round on the filtered data and save the file
        wavcleanEEGforERP=EEG2DforERP-artifacts;
        EEG.data = wavcleanEEGforERP;
        
        EEG = pop_saveset(EEG, 'filename',strrep(FileNames{current_file}, src_file_ext,...
        '_wavcleanforERP.set'),'filepath',[src_folder_name filesep 'intermediate1_wavclean']);

        EEG.setname='wavcleanedEEGforERP';
        EEG.event = events;
        EEG.urevent = orig_event_info;
        
        %apply the IC info from the first processing round to this wavcleanedforERP data
        EEG.icaweights = ICAweightstotransfer;
        EEG.icasphere = ICAspheretotransfer;
        % EEG = eeg_checkset(EEG);
        
        %reject IC components identified as artifact in previous ICA-MARA step:
        EEG.reject.gcompreject = ICs_with_reject_flagged;
        artifact_ICs=find(EEG.reject.gcompreject == 1);
        EEG = pop_subcomp( EEG, artifact_ICs, 0);
        EEG.setname='wavcleanedEEGforERP_ICA_MARA_rej';
        % EEG = eeg_checkset(EEG);
        
        %filter for ERPs:
        EEG = pop_eegfiltnew(EEG,ERP_highpass_cutoff,ERP_lowpass_cutoff,[],0,[],0);
        EEG.setname='wavcleanedEEGforERP_ICA_rej_filt';
        % EEG = eeg_checkset(EEG);
        
        %save the ERP-prepped intermediate file before interpolating anything
        EEG = pop_saveset(EEG, 'filename',strrep(FileNames{current_file}, src_file_ext,'_ICAcleaned_and_filteredERP.set'),'filepath',[src_folder_name filesep 'intermediate2_ICAclean']);
    end
    
    %% segment data according to data type
    if segment_data
        if ~task_EEG_processing
            EEG = eeg_regepochs(EEG,'recurrence',segment_length,'limits',[0 segment_length], 'rmbase', [NaN]);
            Number_trials_before_processing=EEG.trials;
        else
            %first, transform task offset from milliseconds to samples
            samples_offset= srate*task_offset/1000
            %then, correct for timing offset (in samples) between event initiation and presentation
            for i = 1:size(EEG.event,2)
                EEG.event(i).latency = EEG.event(i).latency+samples_offset;
            end
            %then generate segments around the corrected stimulus presentation timing           
            EEG = pop_epoch(EEG, task_onset_tags, [task_segment_start task_segment_end], 'verbose', 'no', 'epochinfo', 'yes');
            Number_trials_before_processing=EEG.trials;
        end
    end
    
    EEG = pop_saveset( EEG, 'filename',strrep(FileNames{current_file},src_file_ext,'_segmented.set'),'filepath',[src_folder_name filesep 'intermediate3_segmented']);
    
    %% baseline correct task EEG if requested
    if task_EEG_processing && exist('baseline_correction', 'var') && baseline_correction
        EEG = pop_rmbase(EEG, [baseline_corr_start   baseline_corr_end]);
        % EEG = eeg_checkset(EEG);
    end
    
    EEG = pop_saveset( EEG, 'filename',strrep(FileNames{current_file},src_file_ext,...
        '_segmented_BLcorrected.set'),'filepath',[src_folder_name filesep 'intermediate3_segmented']);
    
    %% if selected option, interpolate bad data within segments from "good channels" only:
    if segment_interpolation ==1
        
        %use only the good channels to evaluate data:
        eeg_chans=[1:length(selected_good_channel_locations)];
        
        % evaluate the channels for each segment and interpolate channels with bad
        % data for that each segment using the FASTER program, interpolating channels scoring above/below z threshold of 3 for an segment:
        % code taken from FASTER program (Nolan et al., 2010)
        ext_chans=[];
        o.epoch_interp_options.rejection_options.measure = [1 1 1 1];
        o.epoch_interp_options.rejection_options.z = [3 3 3 3];
        
        if  length(size(EEG.data)) > 2
            status = '';
            lengths_ep=cell(1,size(EEG.data,3));
            for v=1:size(EEG.data,3)
                list_properties = single_epoch_channel_properties(EEG,v,eeg_chans);
                lengths_ep{v}=eeg_chans(logical(min_z(list_properties,o.epoch_interp_options.rejection_options)));
                status = [status sprintf('%d: ',v) sprintf('%d ',lengths_ep{v}) sprintf('\n')];
            end
            EEG=h_epoch_interp_spl(EEG,lengths_ep,ext_chans);
            EEG.saved='no';
            
            %add the info about which channels were interpolated for each segment to the EEG file
            EEG.etc.epoch_interp_info=[status];
            Channels_Interpolated_For_Each_Segment(current_file) = cellstr(status);
        end
        
        EEG = pop_saveset(EEG, 'filename',strrep(FileNames{current_file}, src_file_ext,'_segments_interp.set'),'filepath',[src_folder_name filesep 'intermediate3_segmented']);
    else
        Channels_Interpolated_For_Each_Segment(current_file) = {''};
    end
    
    %% rejection of bad segments
    %using amplitude-based and joint probability artifact detection:
    if ~ROI_channels_only
        EEG = pop_eegthresh(EEG,1,[1:EEG.nbchan] ,[reject_min_amp],[reject_max_amp],[EEG.xmin],[EEG.xmax],2,0);
        EEG = pop_jointprob(EEG,1,[1:EEG.nbchan],3,3,pipeline_visualizations_semiautomated,...
            0,pipeline_visualizations_semiautomated,[],pipeline_visualizations_semiautomated);
    else
        EEG = pop_eegthresh(EEG,1,[ROI_indices_in_selected_chanlocs]',[reject_min_amp],[reject_max_amp],[EEG.xmin],[EEG.xmax],2,0);
        EEG = pop_jointprob(EEG,1,[ROI_indices_in_selected_chanlocs]',3,3,pipeline_visualizations_semiautomated,...
            0,pipeline_visualizations_semiautomated,[],pipeline_visualizations_semiautomated);
    end
    
    % reject segments (trials) because user event list has flagged them as bad trials
    if task_EEG_processing == 1 && netstationRAWformat == 0 && user_selected_trials == 1
        EEG = pop_selectevent( EEG, 'status','good','deleteevents','on','deleteepochs','on','invertepochs','off');
        % EEG = eeg_checkset(EEG );
    end
    
    EEG = eeg_rejsuperpose(EEG, 1, 0, 1, 1, 1, 1, 1, 1);
    EEG = pop_rejepoch(EEG, [EEG.reject.rejglobal] ,0);
    % EEG = eeg_checkset(EEG);
    EEG = pop_saveset(EEG, 'filename',strrep(FileNames{current_file}, src_file_ext,...
        '_segments_postreject.set'),'filepath',[src_folder_name filesep 'intermediate3_segmented']);
    
    %% interpolate the channels that were flagged as bad earlier:
    EEG = pop_interp(EEG, full_selected_channels, 'spherical');
    EEG.setname='wavcleanedEEG_ICA_MARA_rej_chan_int';
    % EEG = eeg_checkset(EEG);
    
    %% re-reference the data: average reference used here
    if average_rereference == 1;
        EEG = pop_reref(EEG, []);
        EEG.setname='wavcleanedEEG_ICA_MARA_rej_chan_int_avgreref';
        % EEG = eeg_checkset(EEG);
    else
        [~,ref_chan_indices_in_full_selected_chanlocs] = intersect({full_selected_channels.labels},NO_AVERAGE_REREF_channel_subset,'stable');
        EEG = pop_reref(EEG, ref_chan_indices_in_full_selected_chanlocs);
        EEG.setname='wavcleanedEEG_ICA_MARA_rej_chan_int_chansubsetreref';
        % EEG = eeg_checkset(EEG);
    end
    
    %% store outputs and report metrics
    Number_Channels_User_Selected(current_file)=size(chan_IDs,2);
    Number_ICs_Rejected(current_file)=length(artifact_ICs);
    Number_Good_Channels_Selected(current_file)=size(selected_good_channel_locations,2);
    Percent_Good_Channels_Selected(current_file)=Number_Good_Channels_Selected(current_file)/Number_Channels_User_Selected(current_file)* 100;
    Percent_ICs_Rejected(current_file)=Number_ICs_Rejected(current_file)/Number_Good_Channels_Selected(current_file)* 100;
    Percent_Variance_Kept_of_Post_Waveleted_Data(current_file)=varianceWav;
    if isempty(bad_channels_removed)
        Interpolated_Channel_IDs{current_file} = 'none';
    else
        Interpolated_Channel_IDs{current_file}=[sprintf('%s ',bad_channels_removed{1:end-1}),bad_channels_removed{end}];
    end
    Median_Artifact_Probability_of_Kept_ICs(current_file)=median_artif_prob_good_ICs;
    Mean_Artifact_Probability_of_Kept_ICs(current_file)=mean_artif_prob_good_ICs;
    Range_Artifact_Probability_of_Kept_ICs(current_file)=range_artif_prob_good_ICs;
    Min_Artifact_Probability_of_Kept_ICs(current_file)=min_artif_prob_good_ICs;
    Max_Artifact_Probability_of_Kept_ICs(current_file)=max_artif_prob_good_ICs;
    Number_Segments_Post_Segment_Rejection(current_file)=EEG.trials;
    Number_Segments_BEFORE_segment_rejection(current_file)=Number_trials_before_processing;
    cd ([src_folder_name filesep 'processed']);
    
    %% save preprocessed dataset with subject ID as user specified file type
    switch save_as_format
        case 1 % txt file
            pop_export(EEG,strrep(FileNames{current_file}, src_file_ext,'_processed_IndivTrial.txt'),'transpose','on','precision',8);
            pop_export(EEG,strrep(FileNames{current_file}, src_file_ext,'_processed_AveOverTrials.txt'),'transpose','on','erp','on','precision',8);
        case 2 % .mat file
            save(strrep(FileNames{current_file}, src_file_ext,'_processed.mat'), 'EEG');
        case 3 % .set file
            EEG = pop_saveset(EEG, 'filename',strrep(FileNames{current_file}, src_file_ext,'_processed.set'),'filepath',[src_folder_name filesep 'processed']);
    end
    
    %% generate power spectrum and topoplot visualization if user requested:
    %plot the spectrum across channels to evaluate pipeline performance
    if pipeline_visualizations_semiautomated == 1
        if ERP_analysis == 0;
            figure; pop_spectopo(EEG, 1, [], 'EEG' , 'freq', [[freq_to_plot]], 'freqrange',[[vis_freq_min] [vis_freq_max]],'electrodes','off');
            saveas (gcf,strrep(FileNames{current_file}, src_file_ext,'_processedspectrum.jpg'));
        elseif ERP_analysis == 1;
            figure; pop_timtopo(EEG, [vis_time_start  vis_time_end], [NaN], '');
            saveas (gcf,strrep(FileNames{current_file}, src_file_ext,'_processed_ERP.jpg'));
        end
    end
end

%% generate output table in the "preprocessed" subfolder listing the subject file name and relevant variables for assesssing how good/bad that datafile was and how well the pipeline worked
outputtable=table(FileNames',File_Length_In_Secs',Number_Channels_User_Selected',Number_Segments_BEFORE_segment_rejection',Number_Segments_Post_Segment_Rejection',...
    Number_Good_Channels_Selected', Percent_Good_Channels_Selected', Interpolated_Channel_IDs',Number_ICs_Rejected',...
    Percent_ICs_Rejected', Percent_Variance_Kept_of_Post_Waveleted_Data',Median_Artifact_Probability_of_Kept_ICs',...
    Mean_Artifact_Probability_of_Kept_ICs',Range_Artifact_Probability_of_Kept_ICs',Min_Artifact_Probability_of_Kept_ICs',...
    Max_Artifact_Probability_of_Kept_ICs',Channels_Interpolated_For_Each_Segment');
outputtable.Properties.VariableNames ={'FileNames','File_Length_In_Secs','Number_Channels_User_Selected','Number_Segments_Before_Segment_Rejection','Number_Segments_Post_Segment_Rejection',...
    'Number_Good_Channels_Selected', 'Percent_Good_Channels_Selected', 'Interpolated_Channel_IDs','Number_ICs_Rejected',...
    'Percent_ICs_Rejected', 'Percent_Variance_Kept_of_Post_Waveleted_Data','Median_Artifact_Probability_of_Kept_ICs',...
    'Mean_Artifact_Probability_of_Kept_ICs','Range_Artifact_Probability_of_Kept_ICs','Min_Artifact_Probability_of_Kept_ICs',...
    'Max_Artifact_Probability_of_Kept_ICs','Channels_Interpolated_For_Each_Segment'};

rmpath(genpath(cleanline_path));
writetable(outputtable, ['HAPPE_all_subs_output_table ',datestr(now,'dd-mm-yyyy'),'.csv']);


%% USER INPUT FUNCTIONS
% DETERMINE AQUISITION LAYOUT
function layout_type = determ_aquiLayout()
    while true
        % prompt user for layout and store input
        layout_type = input(['Enter aquisition layout:\n' ...
            '  1 = EGI Hydrocel Geodesic Sensor Net 32 Channel v1.0\n' ...
            '  2 = EGI Geodesic Sensor Net 64 Channel v2.0\n' ...
            '  3 = EGI Hydrocel Geodesic Sensor Net 128 Channel v1.0\n' ...
            'NOTE: For other nets, run through BEAPP, as described in the HAPPE manual\n' ...
            '> ']) ;
        % if valid input, proceed to next user input
        if layout_type == 1 || layout_type == 2 || layout_type == 3
            break ;
        % otherwise, alert user to invalid input
        else
            disp("Invalid input: please enter 1, 2, or 3. Otherwise, please use BEAPP.") ;
        end
    end
end

% DETERMINE IF RESTING STATE OR TASK DATA
function [task_EEG_processing, ERP_analysis] = determ_rVt()
    disp("Enter data type:") ;
    disp("0 = Resting-State EEG") ;
    disp("1 = Task-Related EEG") ;
    while true
        % request and store user input
        task_EEG_processing = input('> ') ;
        % IF TASK DATA...
        if task_EEG_processing
            % DETERMINE IF ERP PROCESSING...
            disp("Performaing event-related potential (ERP) analysis? [Y/N]") ;
            while true
                % request and store user input
                user_input = input('> ', 's') ;
                % IF ERP PROCESSING... 
                if strcmpi(user_input, "y")
                    ERP_analysis = 1 ;
                    break ;
                % IF NOT ERP PROCESSING... continue to next user input
                elseif strcmpi(user_input, "n")
                    ERP_analysis = 0 ;
                    break ; 
                % otherwise, alert user to invalid input
                else
                    disp("Invalid input: please enter Y or N.") ;
                end
            end
            break ;
        % IF REST... continue to next user input
        elseif ~task_EEG_processing
            ERP_analysis = 0 ;
            break ;
        % otherwise, alert user to invalid input
        else
            disp("Invalid input: please enter 0 or 1.") ;
        end
    end
end

% DETERMINE HIGH AND LOW PASS FILTERS:
function [ERP_lowpass_cutoff, ERP_highpass_cutoff] = determ_erpCutoff()
    ERP_lowpass_cutoff = input(['Enter low-pass filter, in Hz:\n' ...
        'Common low-pass filter is 30 - 45 Hz\n' ...
        '> ']) ;
    ERP_highpass_cutoff = input(['Enter high-pass filter, in Hz:\n' ...
        'Common high-pass filter is 0.1 - 0.3 Hz.\n' ...
        '> ']) ;
end

% COMPILE CHANNELS OF INTEREST
function chan_IDs = determ_chanIDs(layout_type)
    disp("Examine all channels (all) or only channels of interest (coi)?") ;
    while true
        % collect and store user input
        user_input = input('> ', 's') ;
        % if the user requests all channels, retrieve list using aqui. layout
        if strcmpi(user_input, 'all')
            if layout_type == 1
                disp("ERROR: No current layout established in netdata_lib for this layout.") ;
            elseif layout_type == 2
                chan_IDs = {'FP1' 'FP2' 'F7' 'F3' 'F4' 'F8' 'C3' 'CZ' 'C4' 'T5' 'PZ' ...
                    'T6' 'O1' 'O2' 'T3' 'T4' 'P3' 'P4' 'Fz' 'E1' 'E2' 'E4' 'E5' 'E7' ...
                    'E8' 'E9' 'E10' 'E12' 'E14' 'E16' 'E18' 'E19' 'E20' 'E21' 'E22' ...
                    'E23' 'E25' 'E26' 'E29' 'E30' 'E31' 'E32' 'E33' 'E35' 'E36' 'E38' ...
                    'E39' 'E41' 'E42' 'E43' 'E44' 'E45' 'E47' 'E48' 'E50' 'E51' 'E53' ...
                    'E55' 'E56' 'E57' 'E58' 'E59' 'E60' 'E63' 'E64'} ;
            elseif layout_type == 3
                chan_IDs = {'FP1' 'FP2' 'F7' 'F3' 'F4' 'F8' 'C3' 'CZ' 'C4' 'T5' 'PZ' ...
                    'T6' 'O1' 'O2' 'T3' 'T4' 'P3' 'P4' 'Fz' 'E1' 'E2' 'E3' 'E4' 'E5' ...
                    'E6' 'E7' 'E8' 'E10' 'E12' 'E13' 'E14' 'E15' 'E16' 'E17' 'E18' 'E19' ...
                    'E20' 'E21' 'E23' 'E25' 'E26' 'E27' 'E28' 'E29' 'E30' 'E31' 'E32' ...
                    'E34' 'E35' 'E37' 'E38' 'E39' 'E40' 'E41' 'E42' 'E43' 'E44' 'E46' ...
                    'E47' 'E48' 'E49' 'E50' 'E51' 'E53' 'E54' 'E55' 'E56' 'E57' 'E59' ...
                    'E60' 'E61' 'E63' 'E64' 'E65' 'E66' 'E67' 'E68' 'E69' 'E71' 'E72' ...
                    'E73' 'E74' 'E75' 'E76' 'E77' 'E78' 'E79' 'E80' 'E81' 'E82' 'E84' ...
                    'E85' 'E86' 'E87' 'E88' 'E89' 'E90' 'E91' 'E93' 'E94' 'E95' 'E97' ...
                    'E98' 'E99' 'E100' 'E101' 'E102' 'E103' 'E105' 'E106' 'E107' 'E109' ...
                    'E110' 'E111' 'E112' 'E113' 'E114' 'E115' 'E116' 'E117' 'E118' 'E119' ...
                    'E120' 'E121' 'E123' 'E125' 'E126' 'E127' 'E128'}  ;
            else
                disp("Invalid input: this layout type is not supported by this version of HAPPE.") ;
            end
            break ;
        % If the user wants a subset, request user input to collect channels of interest
        elseif strcmpi(user_input, "coi")
            disp("Enter channels of interest, one at a time, pressing enter/return between each entry.") ;
            disp("When you have entered all channels, input 'done' (without quotations).") ;
            disp("NOTE: 10-20 channels are already included.") ;
            % set the 10-20 channels
            chan_IDs = {'FP1' 'FP2' 'F3' 'F4' 'F7' 'F8' 'C3' 'C4' 'T3' 'T4' 'PZ' ...
                'O1' 'O2' 'T5' 'T6' 'P3' 'P4' 'Fz'} ;
            % initialize the index for user-entered channels
            indx = 19 ;
            while true
                % request and store user input
                u_input = input('> ', 's') ;
                % if done, stop collecting channels and eliminate duplicates
                if strcmpi(u_input, 'done')
                    chan_IDs = unique(chan_IDs, 'stable') ;
                    break ;
                % add user input channels to cell array of channels
                else
                    chan_IDs{indx} = u_input ;
                    indx = indx + 1 ;
                end
            end
            break ;
        % Otherwise alert user to invalid input
        else
            disp("Invalid input: please enter 'all' or 'coi' (without quotations)") ;
        end
    end
end

% DETERMINE IF NETSTATION .RAW OR MATLAB ARRAY
function netstationRAWformat = determ_fileFormat()
    disp("File format:") ;
    disp("0 = Matlab array") ;
    disp("1 = Netstation simple binary .RAW format") ;
    while true
        % collect and store user input
        netstationRAWformat = input('> ') ;
        % IF MATLAB ARRAY OR .RAW...
        if netstationRAWformat == 0 || netstationRAWformat == 1
            break ;
        % otherwise, alert user to invalid input
        else 
            disp("Invalid input: please enter 0 or 1.") ;
        end
    end
end
% mat-lab specific
function [potential_eeg_var_names, sampling_rate_varname] = fileFormat_matlabArray()
    % DETERMINE POTENTIAL NAMES OF EEG DATA VARIABLE
    disp("Enter potential matlab variable names for your EEG data, one at a time.") ;
    disp("When you have entered all potential names, input 'done' (without quotations).") ;
    disp("NOTE: Variable names including 'Segment' may cause issues, as data should not be pre-epoched.") ;
    indx = 1 ;
    while true
        % collect and store user input
        user_input = input('> ', 's') ;
        % if done, stop collecting and remove duplicates
        if strcmpi(user_input, 'done')
            potential_eeg_var_names = unique(potential_eeg_var_names, 'stable');
            break ;
        % add user input to cell array of potential names    
        else
            potential_eeg_var_names{indx} = user_input ;
            indx = indx + 1 ;
        end
     end
     % DETERMINE SAMPLING RATE - only for matlab arrays***
     sampling_rate_varname = {input(['Sampling rate variable name:\n' ...
        'NOTE: NetStation files v4.x default = samplingRate\n' ...
        '      NetStation files v5.x default = EEGSamplingRate\n' ...
        '> '], 's')} ;
end
function task_event_info_location = fileFormat_matlabArray_txtLocation()
    % DETERMINE LOCATION OF .TXT FILES containing task info
    task_event_info_location = input('Path to .txt files containing task event info:\n> ', 's') ;
end
% .RAW specific
function task_onset_tags = fileFormat_dotRAW()
    % ENTER STIM ONSET TAGS
    disp("Enter stimulus onset tags, one at a time.") ;
    disp("When you have entered all potential tage, input 'done' (without quotations).") ;
    disp("REMEMBER: Mark up your netstation files (segmentation tools --> segment markup) before processing with HAPPE.") ;
    indx = 1 ;
    while true
        % request and store user input
        user_input = input('> ', 's') ;
        % if done, stop collecting and remove duplicates from list
        if strcmpi(user_input, 'done')
            task_onset_tags = unique(task_onset_tags, 'stable') ;
            break ;
        % add user input to cell array of tags
        else
            task_onset_tags{indx} = user_input ;
            indx = indx + 1 ;
        end
    end
end

% DETERMINE IF DOWNSAMPLING
function downsample_data = determ_downsample()
    disp("Downsample data? [Y/N]") ;
    while true
        % collect and store user input
        user_input = input('> ', 's') ;
        if strcmpi(user_input, 'y')
            downsample_data = 1 ;
            break ;
        elseif strcmpi(user_input, 'n')
            downsample_data = 0 ;
            break ;
        else
            disp("Invalid input: please enter Y or N.") ;
        end
    end
end

% DETERMINE IF RUNNING WITH VISUALIZATIONS
function pipeline_visualizations_semiautomated = determ_visualizations()
    disp("Run HAPPE with visualizations? [Y/N]") ;
    while true
        % collect and store user input
        user_input = input('> ', 's') ;
        % IF YES TO VISUALIZATIONS...
        if strcmpi(user_input, "y")
            pipeline_visualizations_semiautomated = 1 ;
            break ;
        % if no to visualizations, continue to next user input    
        elseif strcmpi(user_input, 'n')
            pipeline_visualizations_semiautomated = 0 ;
            break ;
        % otherwise, alert user to invalid input
        else
            disp("Invalid input: please enter Y or N.") ;
        end
    end
end
function [vis_freq_min, vis_freq_max, freq_to_plot] = visulization_param()
    % DETERMINE MIN AND MAX FOR POWER SPECT. VISUALIZATION
    vis_freq_min = input("Minimum value for power spectrum figure:\n> ") ;
    vis_freq_max = input("Maximum value for power spectrum figure:\n> ") ;
            
    % DETERMINE FREQUENCY FOR SPATIAL TOPOPLOTS
    disp("Enter the frequencies, one at a time, to generate spatial topoplots for:") ;
    disp("When you have entered all frequencies, input 'done' (without quotations).") ;
    indx = 1 ;
    while true
        user_input = input('> ', 's') ;
        if strcmpi(user_input, 'done')
            freq_to_plot = unique(freq_to_plot, 'stable') ;
            break ;
        else
            freq_to_plot(indx) = str2num(user_input) ;
            indx = indx + 1 ;
        end
    end
end
function [vis_time_start, vis_time_end] = visualization_param_erp()
    % DETERMINE TIME RANGE FOR THE TIMESERIES FIGURE        
    vis_time_start = input('Start time, in MILLISECONDS, for the ERP timeseries figure:\n> ') ;
    vis_time_end = input(['End time, in MILLISECONDS, for the ERP timeseries figure:\n' ...
        'NOTE: This should end 1 millisecond before your segmentation parameter ends. (e.g. 299 for 300)\n' ...
        '> ']) ;
end

% DETERMINE IF SEGMENTING
function segment_data = determ_segment()
    disp("Segment data? [Y/N]") ;
    while true
        % store user input
        user_input = input('> ', 's') ;
        % IF YES
        if strcmpi(user_input, 'y')
            segment_data = 1 ;
            break ;
        % IF NO
        elseif strcmpi(user_input, "n")
            segment_data = 0 ;
            break ;
        else
            disp("Invalid input: please enter Y or N.") ;
        end
    end
end
function [task_segment_start, task_segment_end] = segment_task()
    % SET SEGMENT START AND END
    task_segment_start = input("Starting parameter, in SECONDS, to segment the data for each stimulus:\n> ") ;
    task_segment_end = input("Ending parameter, in SECONDS, to segment the data for each stimulus:\n> ") ;
end
function [task_offset, baseline_correction] = segment_erp()
    % DETERMINE TASK OFFSET
    task_offset = input(['Offset delay, in MILLISECONDS, between stimulus initiation and presentation:\n' ...
        'NOTE: Please enter the total offset (combined system and task-specific offsets).\n' ...
        '> ']) ;
    % DETERMINE IF WANT BASELINE CORRECTION
    disp("Perform baseline correction (by subtraction)? [Y/N]") ;
    while true
        u_input = input('> ', 's') ;
        if strcmpi(u_input, "y")
            baseline_correction = 1 ;
            break ;
        elseif strcmpi(u_input, "n")
            baseline_correction = 0 ;
            break ;
        else
            disp("Invalid input: please enter Y or N.") ;
        end
    end
end
function [baseline_corr_start, baseline_corr_end] = baseline_corr()
    % DETERMINE BASELINE START AND END
    baseline_corr_start = input("Enter, in MILLISECONDS, where the baseline segment begins:\n> ") ;
    baseline_corr_end = input(['Enter, in MILLISECONDS, where the baseline segment ends:\n' ...
        'NOTE: 0 indicates stimulus onset.\n> ']) ;
end
function segment_length = segment_rest()
    % DETERMINE SEGMENT LENGTH
    segment_length = input("Segment length, in SECONDS:\n> ") ;
end

% DETERMINE IF INTERPOLATING
function segment_interpolation = determ_interpolation()
    disp("Interpolate the specific channels' data determined to be artifact/bad within each segment? [Y/N]") ;
    while true
        user_input = input('> ', 's') ;
        if strcmpi(user_input, "Y")
            segment_interpolation = 1;
            break ;
        elseif strcmpi(user_input, "N")
            segment_interpolation = 0 ;
            break ;
        else
            disp("Invalid input: please enter Y or N.") ;
        end
    end
end

% DETERMINE IF REJECTING SEGMENTS
function segment_rejection = determ_segReject()
    disp("Perform segment rejection (using amplitude and joing probability criteria)? [Y/N]") ;
    while true
        user_input = input('> ', 's') ;
        % IF YES
        if strcmpi(user_input, "y")
            segment_rejection = 1 ;
            break ;
        elseif strcmpi(user_input, "N")
            segment_rejection = 0 ;
            break ;
        else
            disp("Invalid input: please enter Y or N.") ;
        end
    end
end
function [reject_min_amp, reject_max_amp] = segReject_amp()
    reject_min_amp = input(['Minimum signal amplitude to use as the artifact threshold:\n' ...
        'We recommend -40 for time-frequency analyses; -80 for ERP analyses.\n' ...
        '> ']) ;
    reject_max_amp = input(['Maximum signal amplitude to use as the artifact threshold:\n' ...
        'We recommend 40 for time-frequency analyses; 80 for ERP analyses.\n' ...
        '> ']) ;
end
function ROI_channels_only = segReject_ROI()
% DETERMINE IF ALL OR ROI REJECTION
    disp("Use all user-specified channels (all) or a subset of channels in an ROI (roi)?") ;
    while true
        u_input = input('> ', 's') ;
        if strcmpi(u_input, "roi")
            ROI_channels_only = 1 ;
            break ;        
        elseif strcmpi(u_input, 'all')
            ROI_channels_only = 0 ;
            break ;
        else
            disp("Invalid input: please enter all or roi.") ;
        end
    end
end
function ROI_channels = segReject_ROIchannels()
    disp("Enter the channels in the ROI, one at a time.") ;
    disp("When you have finished entering all channels, enter 'done' (without quotations).") ;
    indx = 1 ;
    while true
        user_input = input('> ', 's') ;
        if strcmpi(u_i, 'done')
            ROI_channels = unique(ROI_channels, 'stable') ;
            break ;
        else
            ROI_channels{indx} = u_i ;
            indx = indx + 1 ;
        end
    end
end
function user_selected_trials = segReject_trials()
    % DETERMINE IF USER PRE-SELECTED "USABLE" TRIALS - Only for
    % task & matlab files?***
    disp("Use pre-selected 'usable' trials to restrict analysis? [Y/N]") ;
    while true
        user_input = input('> ', 's') ;
        if strcmpi(user_input, "y")
            user_selected_trials = 1 ;
        elseif strcmpi(user_input, "n")
            user_selected_trials = 0 ;
            break ;
        else
            disp("Invalid input: pleaser enter Y or N.") ;
        end
    end
end

% DETERMINE TYPE OF RE-REFERENCING
function average_rereference = determ_reref()
    disp("Re-Referencing Type:") ;
    disp("  0 = Re-referencing to another channel/subset of channels") ;
    disp("  1 = Average re-referencing") ;
    while true
        average_rereference = input('> ') ;
        if average_rereference == 0
            break ;
        elseif average_rereference == 1
            break ;
        else
            disp("Invalid input: please enter 0 or 1.") ;
        end
    end
end
function NO_AVERAGE_REREF_channel_subset = reref_nonaverage()
    % CHOOSE THE CHANNEL/SUBSET OF CHANNELS
    disp("Enter channel/subset of channels to re-reference to, one at a time.") ;
    disp("When you have entered all channels, input 'done' (without quotations).") ;
    indx = 1 ;
    while true
    	user_input = input('> ', 's') ;
    	if strcmpi(u_i, 'done')
        	NO_AVERAGE_REREF_channel_subset = unique(NO_AVERAGE_REREF_channel_subset, 'stable') ;
        	break ;
        else
        	NO_AVERAGE_REREF_channel_subset{indx} = user_input ;
            indx = indx + 1 ;
        end
    end
end

% DETERMINE SAVE FORMAT FOR PROCESSED DATA
function save_as_format = determ_saveFormat()
    disp("Format to save processed data:") ;
    disp("  1 = .txt file (electrodes as columns, time as rows) - Choose this for ERP timeseries")
    disp("  2 = .mat file (matlab format)") ;
    disp("  3 = .set file (EEGLab format)") ;
    while true
        save_as_format = input('> ') ;
        if save_as_format == 1 || save_as_format == 2 || save_as_format == 3
            break ;
        else
            disp("Invalid input: please enter 1, 2, or 3.") ;
        end
    end
end