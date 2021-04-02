%% _data, _dcf, _traj
%H5fname = '/data/larson4/UTE_Lung/2019-09-23_vo1/UTElrs/MRI_Raw';
%outfname = '/data/larson4/UTE_Lung/2019-09-23_vo1/UTElrs/20190923_vo1';
%H5fname = '/data/larson4/UTE_Lung/2019-09-23_vo2/cfl/P80896/MRI_Raw';
%H5fname = '/data/larson4/UTE_Lung/2020-01-24_vo/cfl/P37888/MRI_Raw';
%H5fname = '/data/larson4/UTE_Lung/2020-02-13_vo/cfl/P18944/MRI_Raw';
%H5fname = '/data/larson4/UTE_Lung/2018-06-05_patient/cfl/P12800/MRI_Raw';

%H5fname = '/data/larson4/UTE_Lung/2020-07-30_vo/cfl/P44544/MRI_Raw';
%H5fname = '/data/larson4/UTE_Lung/2020-08-20_vo/cfl/P56320/MRI_Raw';
%H5fname = '/data/larson4/UTE_Lung/2020-09-14_vo/cfl/P12288/MRI_Raw';
%H5fname = '/data/larson4/UTE_Lung/2020-09-21_vo/cfl/P28672/MRI_Raw';
H5fname = '/data/larson4/UTE_Lung/2020-11-10_vo/cfl/P08704/MRI_Raw';

%H5fname = '/data/larson4/UTE_Lung/2020-07-30_vo/cfl/P48128/MRI_Raw';
%H5fname = '/data/larson4/UTE_Lung/2020-08-20_vo/cfl/P59904/MRI_Raw';
%H5fname = '/data/larson4/UTE_Lung/2020-09-14_vo/cfl/P15872/MRI_Raw';
%H5fname = '/data/larson4/UTE_Lung/2020-09-21_vo/cfl/P32768/MRI_Raw';
%H5fname = '/data/larson4/UTE_Lung/2020-11-10_vo/cfl/P12800/MRI_Raw';
outfname = H5fname;
%H5fname = '/data/larson4/UTE_Lung/2019-09-13_vo/UTElrs/MRI_Raw';
%outfname = '/data/larson4/UTE_Lung/2019-09-13_vo/UTElrs/20190913_vo';

%H5fname = '/data/larson4/UTE_Lung/2018-12-06/tmp/MRI_Raw';
%outfname = '/data/larson4/UTE_Lung/2018-12-06/UTElrs/20181206_vo';

% H5fname = '/data/larson4/UTE_Lung/2019-01-18_vo/NBDB/MRI_Raw';
% outfname = '/data/larson4/UTE_Lung/2019-01-18_vo/UTElrs/20190118_vo';

h5_convert_mTE(H5fname, outfname)

%% _resp
motion_flag = 0; % 0 = bellow, 1 = k0
resp = ute_motion(H5fname,motion_flag);
writecfl([outfname,'_resp'], resp);

%% half data for reproducibility study or half resolution for TTP study
% halfdata(outfname,1) % first half
% halfres(outfname, 0.5) % half resolution
%% data weighting
%ute_data_weighting_nbdb(outfname, 6, resp)
ute_data_weighting_pr_v2(outfname, 12, 0)

%% high res motion resolved recon
system(['maps.sh ' outfname]) % sensitivity maps
command = ['bart pics -p ' outfname '_dcf_prm -i 30 -R T:1024:0:.01 -R W:7:0:0.01 -t ' outfname '_traj_prm ' outfname '_data_prm ' outfname '_maps_pr ' outfname '_pr_rec_v3'];
system(command)
% command = ['bart pics -p ' outfname '_dcf_pr_nb -i 30 -R T:7:0:.01 -R W:7:0:0.01 -t ' outfname '_traj_pr_nb ' outfname '_data_pr_nb ' outfname '_maps ' outfname '_pr_rec_nb'];
% system(command)
% command = ['bart pics -p ' outfname '_dcf_pr_db -i 30 -R T:7:0:.01 -R W:7:0:0.01 -t ' outfname '_traj_pr_db ' outfname '_data_pr_db ' outfname '_maps ' outfname '_pr_rec_db'];
% system(command)

%% readcfl
pr_rec = squeeze(readcfl([outfname '_pr_rec_v3']));
niftiwrite(abs(pr_rec), [outfname '_pr_rec_v3.nii'])

% pr_rec = squeeze(readcfl([outfname '_pr_rec_nb']));
% niftiwrite(abs(pr_rec), [outfname '_pr_rec_nb.nii'])
% 
% pr_rec_db = squeeze(readcfl([outfname '_pr_rec_db']));
% niftiwrite(abs(pr_rec_db), [outfname '_pr_rec_db.nii'])

%% delete unnecessary files
delete_files(outfname)


