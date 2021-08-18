#!/usr/bin/env python
# coding: utf-8

# import dependency packages
import ants
import antspynet
import numpy as np
import ProtonMRILungSegmentation


def seg(img_path, output_file_prefix):

    image = ants.image_read(img_path)
    image_np = image.numpy()
    image_np = image_np / np.amax(image_np)
    
    # initialize
    mask = np.zeros(np.shape(image_np))
    mask_dilate = np.zeros(np.shape(image_np))
    mask_erode = np.zeros(np.shape(image_np))
    mask_close = np.zeros(np.shape(image_np))
    
    for slc in range(np.shape(image_np)[-1]):
        # bias correction & denoise
        print(slc)
        image_slice = ants.from_numpy(image_np[:,:,:,slc], spacing=(2.5,2.5,2.5))
        image_n4 = ants.abp_n4(image_slice, intensity_truncation = (0.15, 0.85, 512))
        #image_resize = ants.resample_image(image_slice, (3.9,3.9,3.9), False, 0) # trained on 3.9mm iso
        #image_n4 = ants.n4_bias_field_correction(image_slice, verbose=True)
        image_denoise = ants.denoise_image(image_n4)
        print('n4 done')
        
        # save denoised image
        denoise_dir = output_file_prefix + 'imgdenoise.nii';
        ants.image_write(image_denoise, denoise_dir)
        #ants.plot(image_denoise)
        
        # run segmentation
        input_file_name = denoise_dir
        template = 'T_template0.nii.gz'
        ProtonMRILungSegmentation.ProtonMRILungSegmentation(input_file_name, output_file_prefix, template)
        
        # load masks
        left_mask_dir = output_file_prefix + 'LeftLungProbability.nii.gz'
        left_mask = ants.image_read(left_mask_dir)
        right_mask_dir = output_file_prefix + 'RightLungProbability.nii.gz'
        right_mask = ants.image_read(right_mask_dir)
        
        # resize back to spacing = 2.5mm iso
        #left_mask = ants.resample_image(left_mask, (208,128,160), True, 0)
        #right_mask = ants.resample_image(right_mask, (208,128,160), True, 0)
        
        # combine masks and morphology dilation
        total_mask = left_mask + right_mask
        left_mask_binary = ants.get_mask(left_mask)
        right_mask_binary = ants.get_mask(right_mask)
        total_mask_binary = left_mask_binary + right_mask_binary
        
        # morphology dilation, erosion, close
        total_mask_MD = ants.iMath_MD(total_mask_binary, radius = 13)
        total_mask_ME = ants.iMath_ME(total_mask_binary, radius = 1)
        total_mask_MC = ants.iMath_MC(total_mask_binary, radius = 3)
        
        mask[:,:,:,slc] = total_mask_binary.numpy()
        mask_dilate[:,:,:,slc] = total_mask_MD.numpy()
        mask_erode[:,:,:,slc] = total_mask_ME.numpy()
        mask_close[:,:,:,slc] = total_mask_MC.numpy()
    
    
    # save 4D mask
    mask_dir = output_file_prefix + "lung_mask.nii"
    mask_ants = ants.from_numpy(1.0*mask, spacing=(2.5,2.5,2.5,1))
    ants.image_write(mask_ants, mask_dir)
    
    mask_dilate_dir = output_file_prefix + "lung_mask_dilate.nii"
    mask_dilate_ants = ants.from_numpy(1.0*mask_dilate, spacing=(2.5,2.5,2.5,1))
    ants.image_write(mask_dilate_ants, mask_dilate_dir)
    
    mask_erode_dir = output_file_prefix + "lung_mask_erode.nii"
    mask_erode_ants = ants.from_numpy(1.0*mask_erode, spacing=(2.5,2.5,2.5,1))
    ants.image_write(mask_erode_ants, mask_erode_dir)
    
    mask_close_dir = output_file_prefix + "lung_mask_close.nii"
    mask_close_ants = ants.from_numpy(1.0*mask_close, spacing=(2.5,2.5,2.5,1))
    ants.image_write(mask_close_ants, mask_close_dir)
    
    # plot
    import sigpy.plot as pl
    pl.ImagePlot(mask * image_np, x = 0, y = 2)   
    pl.ImagePlot(mask_dilate * image_np, x = 0, y = 2) 
    pl.ImagePlot(mask_close * image_np, x = 0, y = 2)


if __name__ == "__main__":
    # load nii file in ants
    #img_path = '/data/larson4/UTE_Lung/2019-09-23_vo1/tmp/UTElrs/20190923_vo1_pr_rec.nii'
    #img_path = '/data/larson4/UTE_Lung/2020-02-13_vo/cfl/P20480/MRI_Raw_pr_rec.nii'
    #img_path = '/data/larson4/UTE_Lung/2020-01-24_vo/cfl/P37888/MRI_Raw_pr_rec_12bins_v2.nii'
    img_path_list = [
        '/data/larson4/UTE_Lung/2020-07-30_vo/cfl/P44544/MRI_Raw_pr_rec.nii',
        '/data/larson4/UTE_Lung/2020-08-20_vo/cfl/P56320/MRI_Raw_pr_rec.nii',
        '/data/larson4/UTE_Lung/2020-09-14_vo/cfl/P12288/MRI_Raw_pr_rec.nii',
        '/data/larson4/UTE_Lung/2020-09-21_vo/cfl/P28672/MRI_Raw_pr_rec.nii',
        '/data/larson4/UTE_Lung/2020-11-10_vo/cfl/P08704/MRI_Raw_pr_rec.nii',
        '/data/larson4/UTE_Lung/2021-03-12_vo/cfl/P86528/MRI_Raw_pr_rec.nii',
        '/data/larson4/UTE_Lung/2020-07-30_vo/cfl/P48128/MRI_Raw_pr_rec.nii',
        '/data/larson4/UTE_Lung/2020-08-20_vo/cfl/P59904/MRI_Raw_pr_rec.nii',
        '/data/larson4/UTE_Lung/2020-09-14_vo/cfl/P15872/MRI_Raw_pr_rec.nii',
        '/data/larson4/UTE_Lung/2020-09-21_vo/cfl/P32768/MRI_Raw_pr_rec.nii',
        '/data/larson4/UTE_Lung/2020-11-10_vo/cfl/P12800/MRI_Raw_pr_rec.nii',
        '/data/larson4/UTE_Lung/2021-03-12_vo/cfl/P90112/MRI_Raw_pr_rec.nii'
    ]
    #img_path = '/data/larson4/UTE_Lung/2021-04-06_ped_patient/cfl/P38400/MRI_Raw_pr_rec_v3.nii'
    
    
    # set output dir
    #output_file_prefix = '/data/larson4/UTE_Lung/2020-02-13_vo/seg/P20480/'
    #output_file_prefix = '/data/larson4/UTE_Lung/2020-01-24_vo/seg/P37888/'
    
    output_file_prefix_list = ['/data/larson4/UTE_Lung/2020-07-30_vo/seg/P44544/',
                               '/data/larson4/UTE_Lung/2020-08-20_vo/seg/P56320/',
                               '/data/larson4/UTE_Lung/2020-09-14_vo/seg/P12288/',
                               '/data/larson4/UTE_Lung/2020-09-21_vo/seg/P28672/',
                               '/data/larson4/UTE_Lung/2020-11-10_vo/seg/P08704/',
                               '/data/larson4/UTE_Lung/2021-03-12_vo/seg/P86528/',
                               '/data/larson4/UTE_Lung/2020-07-30_vo/seg/P48128/',
                               '/data/larson4/UTE_Lung/2020-08-20_vo/seg/P59904/',
                               '/data/larson4/UTE_Lung/2020-09-14_vo/seg/P15872/',
                               '/data/larson4/UTE_Lung/2020-09-21_vo/seg/P32768/',
                               '/data/larson4/UTE_Lung/2020-11-10_vo/seg/P12800/',
                               '/data/larson4/UTE_Lung/2021-03-12_vo/seg/P90112/']
    
    #output_file_prefix = '/data/larson4/UTE_Lung/2021-04-06_ped_patient/seg/P38400/'
    for ind in range(len(img_path_list)):
        seg(img_path_list[ind], output_file_prefix_list[ind])
