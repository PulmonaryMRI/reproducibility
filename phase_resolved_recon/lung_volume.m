
%segfname = '/data/larson4/UTE_Lung/2020-07-30_vo/seg/P44544/lung_mask_close.nii';
%segfname = '/data/larson4/UTE_Lung/2020-07-30_vo/seg/P48128/lung_mask_close.nii';

%segfname = '/data/larson4/UTE_Lung/2020-08-20_vo/seg/P56320/lung_mask_close.nii';
%segfname = '/data/larson4/UTE_Lung/2020-08-20_vo/seg/P59904/lung_mask_close.nii';

%segfname = '/data/larson4/UTE_Lung/2020-09-14_vo/seg/P12288/lung_mask_close.nii';
%segfname = '/data/larson4/UTE_Lung/2020-09-14_vo/seg/P15872/lung_mask_close.nii';

%segfname = '/data/larson4/UTE_Lung/2020-09-21_vo/seg/P28672/lung_mask_close.nii';
%segfname = '/data/larson4/UTE_Lung/2020-09-21_vo/seg/P32768/lung_mask_close.nii';

%segfname = '/data/larson4/UTE_Lung/2020-11-10_vo/seg/P08704/lung_mask_close.nii';
%segfname = '/data/larson4/UTE_Lung/2020-11-10_vo/seg/P12800/lung_mask_close.nii';

%segfname = '/data/larson4/UTE_Lung/2021-03-12_vo/seg/P86528/lung_mask_close.nii';
segfname = '/data/larson4/UTE_Lung/2021-03-12_vo/seg/P90112/lung_mask_close.nii';

lung_mask = niftiread(segfname);
sz = size(lung_mask);
volume = sum(reshape(lung_mask,[],sz(end)),1) * 0.025^3; % in liters

disp(['end-expiration volume: ' num2str(volume(1))])
disp(['end-inspiration volume: ' num2str(volume(7))])