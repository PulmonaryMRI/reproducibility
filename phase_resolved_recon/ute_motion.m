function motion_signal = ute_motion(h5name,motion_flag)
% motion signal extraction
% input:
%   h5name: ute file name
% output:
%   motion_signal: resp. bellow, ksp0 based signal, img base signal

% please set the bart/matlab directory
addpath /working/larson/xzhu/Lung_data/matlab
TR = 4; 

if(motion_flag==0)
    % bellow gating method
    H5_file = [h5name,'.h5'];
    time = h5read(H5_file,'/Gating/TIME_E0');
    %time = h5read(H5_file,'/Gating/time');
    [~,order] = sort(time);
    t_N = length(order);
    
    resp = h5read(H5_file,'/Gating/RESP_E0');
    % resp = h5read(H5_file,'/Gating/resp');
    resp = resp(order);
    
    motion_signal = resp;
end

if(motion_flag==1)
    % k0 gating method
    rdata_file = [h5name,'_data'];
    ksp = readcfl(rdata_file);
    ksp_c = squeeze(ksp(1,1,:,:));
    t_N = size(ksp_c,1);
    
%    win_len = ceil(100/TR);
%    ksp_r = reshape(ksp_c(1:floor(t_N/win_len)*win_len,:),win_len,floor(t_N/win_len),size(ksp_c,2));
%    ksp_r = squeeze(sum(ksp_r,1));
    
    ksp_r = ksp_c;
    [U,S,V ] = svd(ksp_r,'econ');
    
    S = wthresh(S,'s',S(1,1)*.05);
    motion_signal = U*S*V';
    [~, ind_c] = max(std(real(motion_signal),[],1));
    motion_signal = real(motion_signal(:,ind_c));
%    motion_signal = interp1(motion_signal([1,1:end,end]),(1:t_N)'/win_len+1);
    plot(motion_signal);
    flip_flag = (input('Motion signal flip(>=1):')>=1);
    motion_signal = motion_signal.*(1-2*flip_flag);
    % motion_signal = cconv(motion_signal,ones(win_len,1)/win_len,length(motion_signal));
%     % motion correction 
%     noise = h5read(H5_file,'/Kdata/Noise');
%     noise = noise.real+1i*noise.imag;
%     img_file = [h5name,'_img'];
%     img = readcfl(img_file);
%     img0 = sqrt(sum(abs(img.^2),4));
%     figure;
%     subplot(1,2,1);
%     imshow(squeeze(img0(:,:,floor(size(img,3)/2))),[]);
%     subplot(1,2,2);
%     imshow(squeeze(img0(:,floor(size(img,2)/2),:)),[]);
%     A = abs(noise'*noise);
%     tx=input('please input a number for x:');
%     ty=input('please input a number for y:');
%     tz=input('please input a number for z:');
%     center = [tx,ty,tz];
%     sigma = [10,10,10]/(2*.2);
%     
%     % coefficients
%     a = motion_smap(img,center,sigma,A);
%     motion_signal = abs(squeeze(ksp_c)*a);
end

if (motion_flag==2)
    resp_file = input('RESP file');
    scan_offset = 30e3/40;
    resp = textread(resp_file);
    resp = resp(scan_offset+1:end);
    info = h5info(H5_file,'/Kdata/KX_E0');
    ksize = info.Dataspace.Size;
    t = linspace(1,length(resp),ksize(2));
    motion_signal = interp1(resp,t(:));
end
    
