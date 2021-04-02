 function ute_data_weighting_pr_v2(rootfname, hard_Nbins, resp_flag)

    if nargin < 2
    hard_Nbins = 4;
    end

    data = readcfl([rootfname '_data_pr']);
    traj = readcfl([rootfname '_traj_pr']);
    dcf = readcfl([rootfname '_dcf_pr']);  
   
    if resp_flag == 0  % use k0 and bellow data as ref
        k0 = squeeze(double( data(:,1,:,:) ));
        resp = readcfl([rootfname '_resp']); % peak expiration, valley expiration
        resp_ordered_normalized = (resp-mean(resp))/std(resp);
        gate = ute_k0_gate_extract_pr_v2( k0, resp_ordered_normalized);  
%     elseif resp_flag == 1 % use k0 only for gating
%         k0 = squeeze(double( data(:,1,:,:) ));
%         gate = ute_k0_gate_extract_pr(k0);
    elseif resp_flag == 2 % use bellow only for gating
        resp = readcfl([rootfname,'_resp']);
        resp_ordered_normalized = double((resp-mean(resp))/std(resp));
        resp_in = repmat(resp_ordered_normalized,[1, 2]);
        gate = ute_k0_gate_extract_pr_v2(resp_in, resp_ordered_normalized);
    end
    % use the 10-90 percentile 
%     prc = prctile(gate, [5, 95]);
%     prcSlc = gate>prc(1) & gate<prc(2);
%     gate = gate(prcSlc);
%     data = data(:,:,prcSlc,:);
%     traj = traj(:,:,prcSlc);
%     dcf = dcf(:,:,prcSlc);
      
    % filtering
     fr = 2.4 * 1e-3;
%     cutoff = 0.5; % Hz
%     [b,a] = butter(3, cutoff * fr * 2, 'low');
%     gate = filtfilt(b, a, padarray(gate, [50 0], 'symmetric'));
%     gate = gate(51:end-50);
    time = (1:length(gate))' * fr;
    figure, plot(time, gate)
    hold on, plot(time, resp_ordered_normalized), hold off
    % subdivide into small segments using local maxima
    %[pks, locs] = findpeaks(-gate, 'MinPeakDistance', round(2/fr), 'MinPeakHeight', range(-gate)/2 + min(-gate)); % min peak width = 1 sec
    [exp_pks, exp_locs] = findpeaks(gate, 'Annotate','extents'); % min peak width = 1 sec
    [insp_pks, insp_locs, width, proms] = findpeaks(-gate, 'Annotate','extents'); % min peak width = 1 sec
    hold on, scatter(exp_locs * fr, exp_pks, 'o'), hold off
    
    % calculate respiratory rate
    time = length(gate) * fr / 60;
    resp_num = length(exp_pks);
    resp_rate = resp_num / time;
    disp(['respiratory rate : ', num2str(resp_rate)])
    
    minute_points = 1:round(60/fr):length(gate); % resp per minute
    for ind = 1:(length(minute_points)-1)
        rr(ind) = sum( (exp_locs >= minute_points(ind)) & (exp_locs < minute_points(ind+1)));
    end
    disp(['respiratory rate (mean +/- std): ', num2str(mean(rr)),'+/-', num2str(std(rr))])
    
    % exclusion criteria 
    if exp_locs(1)>insp_locs(1)
        insp_pks = insp_pks(2:end);
        insp_locs = insp_locs(2:end);
        width = width(2:end);
        proms = proms(2:end);
    end
    if exp_locs(end)<insp_locs(end)
        insp_pks = insp_pks(1:end-1);
        insp_locs = insp_locs(1:end-1);
        width = width(1:end-1);
        proms = proms(1:end-1);
    end
    % phase fitting for each segment
%     phase = NaN(1, length(gate));
%     for i = 1:length(locs)-1
%         % segment
%         seg = locs(i):locs(i + 1);
%         if (min(-gate(seg)) <= (range(-gate)/3 + min(-gate))) % Min Valley 
%             phase(seg) = linspace(0 + pi/hard_Nbins, 2*pi + pi/hard_Nbins, length(seg));
%             phase(seg) = rem(phase(seg), 2 * pi);
%         end
%     end
     width_prc = prctile(width, [5, 95]);
     proms_prc = prctile(proms, 5);
     
     phase = NaN(1, length(gate));
     for i = 1:length(insp_locs)
         % segment
         seg = exp_locs(i):exp_locs(i + 1);
         if (width(i) > width_prc(1)) & (width(i) > width_prc(1)) & (proms(i) > proms_prc)
             phase(seg) = linspace(0 + pi/hard_Nbins, 2*pi + pi/hard_Nbins, length(seg));
             phase(seg) = rem(phase(seg), 2 * pi);
         end
     end
    % slicing, discard images before the 1st and after the last peak, and
    % not sutisfy min valley constraint
    slc = ~isnan(phase);   
    
    figure, plot((1:length(gate(slc))) * fr, gate(slc))
    ind = (1:length(gate));
    ind_slc = (1:length(gate(slc)));
    [slc_pks,~] = ismember(ind, exp_locs);
    hold on, scatter(ind_slc(slc_pks(slc))*fr,gate(slc_pks&slc),'o'), hold off
    
    figure, scatter(phase(:,slc), gate(slc))

    %% Binning
    if hard_Nbins
    [bin_data, bin_traj, bin_dcf] = ute_binning( phase(:, slc), hard_Nbins, data(:,:,slc,:), traj(:,:,slc), dcf(:,:,slc) );
    writecfl([rootfname, '_data_prm'], bin_data);
    writecfl([rootfname, '_traj_prm'], bin_traj);
    writecfl([rootfname, '_dcf_prm'], sqrt(bin_dcf));% iter requires
    end
    
end