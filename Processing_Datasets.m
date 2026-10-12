%% ==============================================================
% BIOSIGNAL ANALYSIS & MASTER EXCEL GENERATION FOR COGNITIVE SATURATION
% Project: Analysis of Video Game Types on Cognitive Performance & Workload
% Location: Capstone 2nd Attempt / Processing_Datasets.m
% ==============================================================
clear; clc; close all;

%% =========================
% CONFIGURATION & DYNAMIC PATHS
% =========================
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd; 
end

gameemo_dir  = fullfile(script_dir, 'Datasets', 'GAMEEMO');
moba_dir     = fullfile(script_dir, 'Datasets', 'MOBA');
real_rec_dir = fullfile(script_dir, 'Datasets', 'Real_Recordings');

% Plot Export Directories
plot_dir          = fullfile(script_dir, 'Saved_Plots');
gameemo_plot_dir  = fullfile(plot_dir, 'Subjects_GAMEEMO');
moba_plot_dir     = fullfile(plot_dir, 'Subjects_MOBA');
signal_traces_dir = fullfile(plot_dir, 'Signal_Traces');
psd_bargraphs_dir = fullfile(plot_dir, 'PSD_BarGraphs');

if ~exist(plot_dir, 'dir');          mkdir(plot_dir);          end
if ~exist(gameemo_plot_dir, 'dir');  mkdir(gameemo_plot_dir);  end
if ~exist(moba_plot_dir, 'dir');     mkdir(moba_plot_dir);     end
if ~exist(signal_traces_dir, 'dir'); mkdir(signal_traces_dir); end
if ~exist(psd_bargraphs_dir, 'dir'); mkdir(psd_bargraphs_dir); end

master_excel_path = fullfile(script_dir, 'master_eeg_features.xlsx');

% Standardized 14 10-20 Channels
commonCh = {'AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'};
target_fs = 128;        % Unified sampling rate (Hz)
chunk_sec = 20 * 60;   % 20-minute chunk length (1200 seconds)

%% =========================
% START EEGLAB
% =========================
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

%% =========================
% OUTPUT CONTAINERS
% =========================
Results = table();
thetaMat = []; alphaMat = []; betaMat = []; engMat = [];
datasetNames = {}; allEEGMeanSignals = {}; allEEGTime = {}; allLabels = {};

%% ==============================================================
% PART A: MOBA DATASETS (.vhdr / .eeg - Clean 10 Subjects)
% ==============================================================
disp('================ PROCESSING MOBA DATASETS ================');
all_moba_files = dir(fullfile(moba_dir, '*'));

% Filter ONLY valid .vhdr files (excluding .vhdr.txt, .vhdr.bak)
valid_vhdr_idx = [];
for k = 1:length(all_moba_files)
    fname = all_moba_files(k).name;
    if endsWith(fname, '.vhdr', 'IgnoreCase', true) && ~contains(fname, '.txt') && ~contains(fname, '.bak')
        valid_vhdr_idx = [valid_vhdr_idx; k];
    end
end
moba_files = all_moba_files(valid_vhdr_idx);

fprintf('Found %d valid MOBA BrainVision header files.\n\n', length(moba_files));

for f = 1:length(moba_files)
    vhdr_name = moba_files(f).name;
    
    % Parse Subject ID (e.g., sub-01 -> MOBA_S01)
    tokens = regexp(vhdr_name, 'sub-(\d+)', 'tokens');
    if ~isempty(tokens)
        subj_num = str2double(tokens{1}{1});
        subj_id = sprintf('MOBA_S%02d', subj_num);
    else
        subj_id = sprintf('MOBA_S%02d', f);
    end
    
    % Load BrainVision EEG Dataset
    try
        EEG = pop_loadbv(moba_dir, vhdr_name);
    catch ME
        fprintf('Error loading %s: %s\n', vhdr_name, ME.message);
        continue;
    end
    
    % Channel Matching & Resampling for Full Subject Session
    EEG = pop_select(EEG, 'channel', commonCh);
    if EEG.srate ~= target_fs
        EEG = pop_resample(EEG, target_fs);
    end
    
    % 1. Save Stacked Signal Trace for entire continuous MOBA recording (1 per subject)
    save_stacked_signal_trace(EEG, commonCh, subj_id, 'MOBA', signal_traces_dir);
    
    % Compute Total 20-Minute Chunks
    chunk_samples = chunk_sec * EEG.srate;
    num_chunks = floor(EEG.pnts / chunk_samples);
    
    fprintf('Loaded [%s] -> Subject: %s | Duration: %.2f min | Chunks: %d\n', ...
        vhdr_name, subj_id, (EEG.pnts/EEG.srate)/60, num_chunks);
    
    subj_feats = [];
    
    for c = 1:num_chunks
        start_pnt = (c - 1) * chunk_samples + 1;
        end_pnt   = c * chunk_samples;
        
        % Slice 20-min EEG chunk
        EEG_chunk = pop_select(EEG, 'point', [start_pnt, end_pnt]);
        EEG_chunk = eeg_checkset(EEG_chunk);
        
        chunkLabel = sprintf('%s_C%02d', subj_id, c);
        game_label = 'MOBA_Match';
        
        % Workload level progression across continuous match chunks
        if c <= 2
            emotion_class = 'Low_Workload';
        elseif c <= 5
            emotion_class = 'Optimal_Workload';
        else
            emotion_class = 'Saturated_Fatigue';
        end
        
        % Extract features for master dataset without calling chunk-level plotting
        feat = extract_common_features(EEG_chunk);
        datasetNames{end+1} = chunkLabel; %#ok<AGROW>
        
        thetaMat = [thetaMat; feat.rel_theta']; %#ok<AGROW>
        alphaMat = [alphaMat; feat.rel_alpha']; %#ok<AGROW>
        betaMat  = [betaMat;  feat.rel_beta']; %#ok<AGROW>
        engMat   = [engMat;   feat.engagement']; %#ok<AGROW>
        
        Results = add_result_row(Results, chunkLabel, subj_id, 'MOBA', game_label, emotion_class, c, feat);
        
        t = (0:size(EEG_chunk.data,2)-1) / EEG_chunk.srate;
        meanSignal = mean(double(EEG_chunk.data), 1);
        allEEGMeanSignals{end+1} = meanSignal; %#ok<AGROW>
        allEEGTime{end+1} = t; %#ok<AGROW>
        allLabels{end+1} = chunkLabel; %#ok<AGROW>
        
        fprintf('Processed [%s] | Subj: %s | Type: MOBA | Game: %s | Emotion: %s | Chunk: %d | TBR: %.4f | Eng: %.4f\n', ...
            chunkLabel, subj_id, game_label, emotion_class, c, feat.MeanThetaBeta, feat.MeanEngagement);
        
        subj_feats = [subj_feats; feat]; %#ok<AGROW>
    end
    
    % 2. Save Subject-Wise PSD Bar Graph for Full Session (1 per subject)
    full_session_feat = extract_common_features(EEG);
    save_psd_bargraph(full_session_feat, commonCh, subj_id, 'MOBA', psd_bargraphs_dir);
    
    % 3. Save Single Merged Subject Summary Plot across all chunks (1 per subject)
    if ~isempty(subj_feats)
        save_moba_subject_plot(subj_feats, commonCh, subj_id, moba_plot_dir);
    end
end

%% ==============================================================
% PART B: GAMEEMO DATASETS (.mat - 28 Subjects x 4 Games = 112 Files)
% ==============================================================
disp(' ');
disp('================ PROCESSING GAMEEMO DATASETS ================');
gameemo_files = dir(fullfile(gameemo_dir, 'S*G*AllRawChannels.mat'));

fprintf('Found %d valid GAMEEMO files.\n\n', length(gameemo_files));

% Map GAMEEMO files by Subject to produce subject-level stacked trace & PSD graph
subj_map = containers.Map();
for f = 1:length(gameemo_files)
    fname = gameemo_files(f).name;
    tokens = regexp(fname, '(S\d+)(G\d+)', 'tokens');
    if ~isempty(tokens)
        sid = tokens{1}{1};
        if isKey(subj_map, sid)
            subj_map(sid) = [subj_map(sid); {fullfile(gameemo_files(f).folder, fname)}];
        else
            subj_map(sid) = {fullfile(gameemo_files(f).folder, fname)};
        end
    end
end

subj_keys = keys(subj_map);
for s = 1:length(subj_keys)
    subj_id = subj_keys{s};
    files_list = subj_map(subj_id);
    subj_feats = [];
    combined_x = [];
    
    for g = 1:length(files_list)
        file_path = files_list{g};
        [~, raw_name, ~] = fileparts(file_path);
        
        tokens = regexp(raw_name, '(S\d+)(G\d+)', 'tokens');
        if ~isempty(tokens)
            game_code = tokens{1}{2};
        else
            game_code = 'G1';
        end
        
        % Map GAMEEMO game code to GameType and EmotionClass
        switch game_code
            case 'G1'
                game_label = 'Boring_Game';
                emotion_class = 'Low_Arousal_Boredom';
            case 'G2'
                game_label = 'Calm_Game';
                emotion_class = 'Low_Arousal_Calm';
            case 'G3'
                game_label = 'Horror_Game';
                emotion_class = 'High_Arousal_Stress';
            case 'G4'
                game_label = 'Action_Game';
                emotion_class = 'High_Arousal_Excitement';
            otherwise
                game_label = 'Unknown';
                emotion_class = 'Neutral';
        end
        
        S = load(file_path);
        X = zeros(length(commonCh), length(S.AF3));
        for i = 1:length(commonCh)
            X(i,:) = S.(commonCh{i});
        end
        
        combined_x = [combined_x, X]; %#ok<AGROW>
        
        EEG = build_gameemo_eeg(X, target_fs, commonCh);
        datasetLabel = sprintf('%s_%s', subj_id, game_code);
        
        [Results, thetaMat, alphaMat, betaMat, engMat, datasetNames, allEEGMeanSignals, allEEGTime, allLabels, feat] = ...
            run_full_dataset_pipeline(EEG, datasetLabel, commonCh, Results, ...
            thetaMat, alphaMat, betaMat, engMat, datasetNames, allEEGMeanSignals, allEEGTime, allLabels, subj_id, 'GAMEEMO', game_label, emotion_class, 1, gameemo_plot_dir);
            
        subj_feats = [subj_feats; feat]; %#ok<AGROW>
    end
    
    % Save Stacked Signal Trace and PSD Bar Graph for GAMEEMO Subject
    EEG_comb = build_gameemo_eeg(combined_x, target_fs, commonCh);
    save_stacked_signal_trace(EEG_comb, commonCh, subj_id, 'GAMEEMO', signal_traces_dir);
    if ~isempty(subj_feats)
        save_psd_bargraph(subj_feats(1), commonCh, subj_id, 'GAMEEMO', psd_bargraphs_dir);
    end
end

%% ==============================================================
% PART C: REAL LAB RECORDINGS (Future Laboratory Recordings)
% ==============================================================
disp(' ');
disp('================ PROCESSING REAL LAB RECORDINGS ================');
real_files_vhdr = dir(fullfile(real_rec_dir, '*.vhdr'));

if isempty(real_files_vhdr)
    disp('No lab recordings found in Real_Recordings folder (Skipped for now).');
else
    for f = 1:length(real_files_vhdr)
        vhdr_name = real_files_vhdr(f).name;
        subj_id = sprintf('LAB_S%02d', f);
        
        EEG = pop_loadbv(real_rec_dir, vhdr_name);
        EEG = pop_select(EEG, 'channel', commonCh);
        if EEG.srate ~= target_fs; EEG = pop_resample(EEG, target_fs); end
        
        chunk_samples = chunk_sec * EEG.srate;
        num_chunks = max(1, floor(EEG.pnts / chunk_samples));
        
        for c = 1:num_chunks
            start_pnt = (c - 1) * chunk_samples + 1;
            end_pnt   = min(c * chunk_samples, EEG.pnts);
            EEG_chunk = pop_select(EEG, 'point', [start_pnt, end_pnt]);
            chunkLabel = sprintf('%s_C%02d', subj_id, c);
            
            [Results, thetaMat, alphaMat, betaMat, engMat, datasetNames, allEEGMeanSignals, allEEGTime, allLabels, ~] = ...
                run_full_dataset_pipeline(EEG_chunk, chunkLabel, commonCh, Results, ...
                thetaMat, alphaMat, betaMat, engMat, datasetNames, allEEGMeanSignals, allEEGTime, allLabels, subj_id, 'Real_Lab', 'Lab_Task', 'Lab_Workload', c, plot_dir);
        end
    end
end

%% ==============================================================
% SAVE MASTER EXCEL WORKBOOK
% ==============================================================
writetable(Results, master_excel_path, 'FileType', 'spreadsheet');
fprintf('\n====================================================\n');
fprintf('Master Excel successfully exported to:\n%s\n', master_excel_path);
fprintf('Total Processed Sessions/Chunks: %d\n', height(Results));
fprintf('====================================================\n');

%% ==============================================================
% COMMAND WINDOW GROUP SUMMARY
% ==============================================================
disp('================ GROUP SUMMARY BY DATASET TYPE ================')
disp(groupsummary(Results, 'DatasetType', 'mean', {'MeanTheta','MeanAlpha','MeanBeta','MeanThetaBeta','MeanEngagement','EngagementVar'}))

%% ==============================================================
% OVERVIEW FIGURE: ALL EEG RECORDINGS COMBINED (AUTOSAVED)
% ==============================================================
fig_overview = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1400 650]);
hold on
cmap = lines(max(1, length(allEEGMeanSignals)));
for i = 1:length(allEEGMeanSignals)
    plot(allEEGTime{i}, allEEGMeanSignals{i}, 'Color', cmap(i,:), 'LineWidth', 0.8);
end
xlabel('Time (s)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Mean EEG Amplitude (\muV)', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('Combined EEG Overview across Common 14 Channels (%d Datasets/Chunks)', length(allEEGMeanSignals)), 'FontSize', 13);
grid on
hold off

saveas(fig_overview, fullfile(plot_dir, 'Combined_EEG_Overview.png'));
close(fig_overview);

%% ==============================================================
% PIPELINE & FEATURE EXTRACTION FUNCTIONS
% ==============================================================
function [Results, thetaMat, alphaMat, betaMat, engMat, datasetNames, allEEGMeanSignals, allEEGTime, allLabels, feat] = ...
    run_full_dataset_pipeline(EEG, datasetLabel, commonCh, Results, thetaMat, alphaMat, betaMat, engMat, datasetNames, allEEGMeanSignals, allEEGTime, allLabels, SubjectID, DatasetType, GameType, EmotionClass, ChunkID, save_folder)
    
    feat = extract_common_features(EEG);
    datasetNames{end+1} = datasetLabel;
    
    thetaMat = [thetaMat; feat.rel_theta'];
    alphaMat = [alphaMat; feat.rel_alpha'];
    betaMat  = [betaMat;  feat.rel_beta'];
    engMat   = [engMat;   feat.engagement'];
    
    Results = add_result_row(Results, datasetLabel, SubjectID, DatasetType, GameType, EmotionClass, ChunkID, feat);
    
    t = (0:size(EEG.data,2)-1) / EEG.srate;
    meanSignal = mean(double(EEG.data), 1);
    allEEGMeanSignals{end+1} = meanSignal;
    allEEGTime{end+1} = t;
    allLabels{end+1} = datasetLabel;
    
    fprintf('Processed [%s] | Subj: %s | Type: %s | Game: %s | Emotion: %s | Chunk: %d | TBR: %.4f | Eng: %.4f\n', ...
        datasetLabel, SubjectID, DatasetType, GameType, EmotionClass, ChunkID, feat.MeanThetaBeta, feat.MeanEngagement);
        
    % Generate per-dataset descriptive plot (Autosaved in background)
    plot_dataset_summary_descriptive(feat, commonCh, datasetLabel, save_folder);
end

function EEG = build_gameemo_eeg(X, fs, commonCh)
    X = double(X);
    if size(X,1) ~= length(commonCh) && size(X,2) == length(commonCh)
        X = X';
    end
    EEG = eeg_emptyset;
    EEG.data   = X;
    EEG.nbchan = size(X,1);
    EEG.pnts   = size(X,2);
    EEG.trials = 1;
    EEG.srate  = fs;
    EEG.xmin   = 0;
    EEG.xmax   = (EEG.pnts - 1) / fs;
    for k = 1:length(commonCh)
        EEG.chanlocs(k).labels = commonCh{k};
    end
    EEG = eeg_checkset(EEG);
end

function feat = extract_common_features(EEG)
    bands.theta = [4 8];
    bands.alpha = [8 13];
    bands.beta  = [13 30];
    bands.gamma = [30 45];
    bands.total = [4 45];
    nCh = EEG.nbchan;
    
    feat.channel_labels = string({EEG.chanlocs.labels});
    feat.rel_theta = zeros(nCh,1);
    feat.rel_alpha = zeros(nCh,1);
    feat.rel_beta  = zeros(nCh,1);
    feat.rel_gamma = zeros(nCh,1);
    feat.theta_beta_ratio = zeros(nCh,1);
    feat.engagement = zeros(nCh,1);
    
    feat.hjorth_act = zeros(nCh,1);
    feat.hjorth_mob = zeros(nCh,1);
    feat.hjorth_com = zeros(nCh,1);
    
    for ch = 1:nCh
        x = double(EEG.data(ch,:));
        
        % Spectral Band Power (Welch Method)
        [pxx, f] = pwelch(x, [], [], [], EEG.srate);
        totalPow = bandpower_from_psd(pxx, f, bands.total);
        thetaPow = bandpower_from_psd(pxx, f, bands.theta);
        alphaPow = bandpower_from_psd(pxx, f, bands.alpha);
        betaPow  = bandpower_from_psd(pxx, f, bands.beta);
        gammaPow = bandpower_from_psd(pxx, f, bands.gamma);
        
        feat.rel_theta(ch) = thetaPow / (totalPow + eps);
        feat.rel_alpha(ch) = alphaPow / (totalPow + eps);
        feat.rel_beta(ch)  = betaPow  / (totalPow + eps);
        feat.rel_gamma(ch) = gammaPow / (totalPow + eps);
        feat.theta_beta_ratio(ch) = thetaPow / (betaPow + eps);
        feat.engagement(ch) = betaPow / (alphaPow + thetaPow + eps);
        
        % Time Domain: Hjorth Parameters
        dx = diff(x);
        ddx = diff(dx);
        var_x = var(x);
        var_dx = var(dx);
        var_ddx = var(ddx);
        
        feat.hjorth_act(ch) = var_x;
        feat.hjorth_mob(ch) = sqrt(var_dx / (var_x + eps));
        feat.hjorth_com(ch) = sqrt(var_ddx / (var_dx + eps)) / (feat.hjorth_mob(ch) + eps);
    end
    
    % Frontal Alpha Asymmetry (F4 vs F3)
    iF3 = find(strcmpi(feat.channel_labels, 'F3'));
    iF4 = find(strcmpi(feat.channel_labels, 'F4'));
    if ~isempty(iF3) && ~isempty(iF4)
        feat.FAA = log(feat.rel_alpha(iF4) + eps) - log(feat.rel_alpha(iF3) + eps);
    else
        feat.FAA = NaN;
    end
    
    % Temporal Engagement Trajectory (10-second windows)
    winSec = 10;
    winSamples = winSec * EEG.srate;
    nWin = floor(EEG.pnts / winSamples);
    engagement_ts = zeros(nWin,1);
    
    for w = 1:nWin
        idx = (w-1)*winSamples + 1 : w*winSamples;
        tmp = zeros(nCh,1);
        for ch = 1:nCh
            x = double(EEG.data(ch, idx));
            [pxx, f] = pwelch(x, [], [], [], EEG.srate);
            thetaPow = bandpower_from_psd(pxx, f, bands.theta);
            alphaPow = bandpower_from_psd(pxx, f, bands.alpha);
            betaPow  = bandpower_from_psd(pxx, f, bands.beta);
            tmp(ch) = betaPow / (alphaPow + thetaPow + eps);
        end
        engagement_ts(w) = mean(tmp);
    end
    
    feat.engagement_ts = engagement_ts;
    feat.engagement_var = var(engagement_ts);
    feat.MeanTheta = mean(feat.rel_theta);
    feat.MeanAlpha = mean(feat.rel_alpha);
    feat.MeanBeta  = mean(feat.rel_beta);
    feat.MeanGamma = mean(feat.rel_gamma);
    feat.MeanThetaBeta = mean(feat.theta_beta_ratio);
    feat.MeanEngagement = mean(feat.engagement);
    
    feat.MeanHjorthActivity = mean(feat.hjorth_act);
    feat.MeanHjorthMobility = mean(feat.hjorth_mob);
    feat.MeanHjorthComplexity = mean(feat.hjorth_com);
end

function bp = bandpower_from_psd(pxx, f, frange)
    idx = f >= frange(1) & f < frange(2);
    bp = trapz(f(idx), pxx(idx));
end

function Results = add_result_row(Results, datasetName, SubjectID, DatasetType, GameType, EmotionClass, ChunkID, feat)
    row = table( ...
        string(datasetName), ...
        string(SubjectID), ...
        string(DatasetType), ...
        string(GameType), ...
        string(EmotionClass), ...
        uint8(ChunkID), ...
        feat.MeanTheta, ...
        feat.MeanAlpha, ...
        feat.MeanBeta, ...
        feat.MeanGamma, ...
        feat.MeanThetaBeta, ...
        feat.FAA, ...
        feat.MeanEngagement, ...
        feat.engagement_var, ...
        feat.MeanHjorthActivity, ...
        feat.MeanHjorthMobility, ...
        feat.MeanHjorthComplexity, ...
        'VariableNames', {'Dataset','SubjectID','DatasetType','GameType','EmotionClass','ChunkID','MeanTheta','MeanAlpha','MeanBeta','MeanGamma','MeanThetaBeta','FAA','MeanEngagement','EngagementVar','MeanHjorthActivity','MeanHjorthMobility','MeanHjorthComplexity'} ...
    );
    Results = [Results; row];
end

%% ==============================================================
% AUTOSAVING PLOT FUNCTIONS
% ==============================================================
function save_stacked_signal_trace(EEG, channel_names, subj_id, dataset_type, save_folder)
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1400 800]);
    nCh = EEG.nbchan;
    data = double(EEG.data);
    t = (0:EEG.pnts-1) / EEG.srate;
    
    spacing = 3 * std(data(:));
    if spacing == 0; spacing = 50; end
    
    hold on;
    cmap = jet(nCh);
    yticks_val = zeros(nCh, 1);
    
    for ch = 1:nCh
        offset = (ch - 1) * spacing;
        plot(t, data(ch, :) + offset, 'Color', cmap(ch, :), 'LineWidth', 0.7);
        yticks_val(ch) = offset;
    end
    
    xlabel('Time (seconds)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('EEG Channels (Stacked Offset \muV)', 'FontSize', 11, 'FontWeight', 'bold');
    title(sprintf('Stacked 14-Channel EEG Raw Signals - Subject %s (%s)', subj_id, dataset_type), 'FontSize', 13, 'FontWeight', 'bold');
    set(gca, 'YTick', yticks_val, 'YTickLabel', channel_names, 'FontSize', 9);
    xlim([0, min(120, max(t))]);
    grid on;
    hold off;
    
    out_file = fullfile(save_folder, sprintf('%s_%s_SignalTraces.png', dataset_type, subj_id));
    saveas(fig, out_file);
    close(fig);
end

function save_psd_bargraph(feat, channel_names, subj_id, dataset_type, save_folder)
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 650]);
    psd_data = [feat.rel_theta, feat.rel_alpha, feat.rel_beta, feat.rel_gamma];
    b = bar(psd_data, 'grouped');
    
    b(1).FaceColor = [0.2 0.6 0.8];
    b(2).FaceColor = [0.2 0.8 0.4];
    b(3).FaceColor = [0.9 0.6 0.2];
    b(4).FaceColor = [0.8 0.2 0.2];
    
    title(sprintf('14-Channel Relative Power Spectral Density (PSD) - Subject %s (%s)', subj_id, dataset_type), 'FontSize', 13, 'FontWeight', 'bold');
    xlabel('10-20 Electrodes / Channels', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Relative Power Density (\Sigma Power = 1.0)', 'FontSize', 11, 'FontWeight', 'bold');
    set(gca, 'XTick', 1:length(channel_names), 'XTickLabel', channel_names, 'FontSize', 10);
    xtickangle(45);
    legend({'Theta (4-8 Hz)', 'Alpha (8-13 Hz)', 'Beta (13-30 Hz)', 'Gamma (30-45 Hz)'}, 'Location', 'northeastoutside', 'FontSize', 10);
    grid on;
    
    out_file = fullfile(save_folder, sprintf('%s_%s_PSD_BarGraph.png', dataset_type, subj_id));
    saveas(fig, out_file);
    close(fig);
end

function plot_dataset_summary_descriptive(feat, channel_names, dataset_name, save_folder)
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 700]);
    tl = tiledlayout(2,2, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    % Subplot 1: Relative Band Powers
    nexttile
    bar([feat.rel_theta feat.rel_alpha feat.rel_beta feat.rel_gamma], 'grouped');
    title('Relative Band Power Spectrum', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Relative Power', 'FontSize', 10);
    set(gca, 'XTick', 1:length(channel_names), 'XTickLabel', channel_names);
    xtickangle(45);
    legend({'Theta (\theta)','Alpha (\alpha)','Beta (\beta)','Gamma (\gamma)'}, 'Location', 'best');
    grid on
    
    % Subplot 2: Theta/Beta Ratio (Mental Fatigue Marker)
    nexttile
    bar(feat.theta_beta_ratio, 'FaceColor', [0.2 0.6 0.4]);
    title('Theta / Beta Ratio (TBR)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Ratio', 'FontSize', 10);
    set(gca, 'XTick', 1:length(channel_names), 'XTickLabel', channel_names);
    xtickangle(45);
    grid on
    
    % Subplot 3: Hjorth Parameters
    nexttile
    bar([feat.hjorth_mob feat.hjorth_com], 'grouped');
    title('Hjorth Parameters (Mobility & Complexity)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Parameter Value', 'FontSize', 10);
    set(gca, 'XTick', 1:length(channel_names), 'XTickLabel', channel_names);
    xtickangle(45);
    legend({'Mobility','Complexity'}, 'Location', 'best');
    grid on
    
    % Subplot 4: Temporal Engagement Trajectory & Frontal Alpha Asymmetry (FAA)
    nexttile
    yyaxis left
    plot(feat.engagement_ts, '-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'Color', [0.2 0.4 0.8]);
    ylabel('Engagement Index over Time', 'FontSize', 10);
    grid on
    yyaxis right
    bar(1, feat.FAA, 0.3, 'FaceColor', [0.8 0.4 0.4]);
    ylabel('Frontal Alpha Asymmetry (FAA)', 'FontSize', 10);
    set(gca, 'XTick', 1, 'XTickLabel', {'FAA (F4 - F3)'});
    title('Dynamic Engagement Trajectory & FAA', 'FontSize', 11, 'FontWeight', 'bold');
    
    title(tl, sprintf('%s - Descriptive Summary Features', dataset_name), 'FontSize', 13, 'FontWeight', 'bold');
    
    out_file = fullfile(save_folder, sprintf('%s_Descriptive.png', dataset_name));
    saveas(fig, out_file);
    close(fig);
end

function save_moba_subject_plot(subj_feats, channel_names, subj_id, save_folder)
    fig = figure('Visible','off','Color','w','Position',[100 100 1200 700]);
    tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
    
    nChunks = length(subj_feats); 
    chunk_labels = arrayfun(@(c) sprintf('Chunk %d', c), 1:nChunks, 'UniformOutput', false);
    
    nexttile; mean_tbr = arrayfun(@(s) s.MeanThetaBeta, subj_feats);
    plot(1:nChunks, mean_tbr, '-o', 'LineWidth', 2, 'Color', [0.8 0.2 0.2]); title('Mental Fatigue Progression (Mean TBR)'); xlabel('20-Min Chunks'); ylabel('Mean TBR'); set(gca, 'XTick', 1:nChunks, 'XTickLabel', chunk_labels); grid on;
    
    nexttile; mean_eng = arrayfun(@(s) s.MeanEngagement, subj_feats);
    plot(1:nChunks, mean_eng, '-s', 'LineWidth', 2, 'Color', [0.2 0.4 0.8]); title('Cognitive Engagement Trajectory'); xlabel('20-Min Chunks'); ylabel('Mean Engagement'); set(gca, 'XTick', 1:nChunks, 'XTickLabel', chunk_labels); grid on;
    
    nexttile; h_act = arrayfun(@(s) s.MeanHjorthActivity, subj_feats); h_com = arrayfun(@(s) s.MeanHjorthComplexity, subj_feats);
    yyaxis left; plot(1:nChunks, h_act, '-^', 'LineWidth', 1.5); ylabel('Activity (Power)'); yyaxis right; plot(1:nChunks, h_com, '-d', 'LineWidth', 1.5); ylabel('Complexity'); title('Hjorth Time-Domain Dynamics'); xlabel('20-Min Chunks'); set(gca, 'XTick', 1:nChunks, 'XTickLabel', chunk_labels); grid on;
    
    nexttile; faa_vals = arrayfun(@(s) s.FAA, subj_feats);
    bar(1:nChunks, faa_vals, 0.5, 'FaceColor', [0.3 0.6 0.4]); title('Frontal Alpha Asymmetry (Emotional Valence)'); xlabel('20-Min Chunks'); ylabel('FAA (F4 - F3)'); set(gca, 'XTick', 1:nChunks, 'XTickLabel', chunk_labels); grid on;
    
    title(tl, sprintf('Subject %s (MOBA) - Continuous Workload & Saturation Analysis', subj_id), 'FontSize', 13, 'FontWeight', 'bold');
    out_file = fullfile(save_folder, sprintf('%s_Summary.png', subj_id)); 
    saveas(fig, out_file); 
    close(fig);
end