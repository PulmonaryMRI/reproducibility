function pr_recon(H5fname)
%% _data, _dcf, _traj
outfname = H5fname;
h5_convert_mTE(H5fname, outfname, 1, 1000, [1,1,1])

%% _resp
motion_flag = 0; % 0 = bellow, 1 = k0
resp = ute_motion(H5fname,motion_flag);
writecfl([outfname,'_resp'], resp);

%% data weighting
ute_data_weighting_pr_v2(outfname, 12, 0)

%% high res motion resolved recon
system(['maps.sh ' outfname]) % sensitivity maps
command = ['bart pics -p ' outfname '_dcf_prm -i 30 -R T:1024:0:0.01 -R W:7:0:0.01 -t ' outfname '_traj_prm ' outfname '_data_prm ' outfname '_maps_pr ' outfname '_pr_rec'];
system(command)

%% readcfl
pr_rec = squeeze(readcfl([outfname '_pr_rec']));
niftiwrite(abs(pr_rec), [outfname '_pr_rec.nii'])

%% delete unnecessary files
delete_files(outfname)

end