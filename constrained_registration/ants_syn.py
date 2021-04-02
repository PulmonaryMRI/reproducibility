#!/usr/bin/env python
# coding: utf-8

import ants
import sigpy.plot as pl
import numpy as np
import matplotlib.pyplot as plt
import sys

def ants_reg(image_path, output_dir, mask_path_dilate):

    image = ants.image_read(image_path)
    mask_all = ants.image_read(mask_path_dilate)
    # SyN registration
    # initialization
    image_shape = np.shape(image.numpy())
    result_4d = np.zeros(image_shape)
    JacDet_4d = np.zeros(image_shape)
    deformationField_4d = np.zeros(image_shape + (3,))
    transform = []
    # neighboring registration
    for phase in range(image_shape[-1]):
        fixed = ants.from_numpy(image[:,:,:,phase])
        moving = ants.from_numpy(image[:,:,:,(phase + 1) % image_shape[-1]])
        mask = ants.from_numpy(mask_all[:,:,:,phase])
        # SyN registration 
        # warpedmovout: Moving image warped to space of fixed image. 
        # warpedfixout: Fixed image warped to space of moving image.
        # fwdtransforms: Transforms to move from moving to fixed image.
        # invtransforms: Transforms to move from fixed to moving image.
        regDict = ants.registration(fixed, moving, mask=mask, type_of_transform='SyNOnly', reg_iterations=(100,70,50,20,0))
        transform = transform + regDict["fwdtransforms"] 
    
    # composite transforms
    for phase in range(image_shape[-1]):
        fixed = ants.from_numpy(image[:,:,:,0])
        moving = ants.from_numpy(image[:,:,:,(phase + 1) % image_shape[-1]])
        # apply transforms result image
        result = ants.apply_transforms(fixed, moving, transformlist = transform[0:(phase+1)*2])
        result_4d[:,:,:,(phase+1)%image_shape[-1]] = result.numpy()
        # apply transforms deformation field
        composite = ants.apply_transforms(fixed, moving, transformlist = transform[0:(phase+1)*2], compose='.')
        # jacobian determinant
        reg_jac = ants.create_jacobian_determinant_image(fixed, composite)
        JacDet_4d[:,:,:,(phase+1)%image_shape[-1]] = reg_jac.numpy()
       
    # save registered image and jacobian determinant 
    result_4d_ants = ants.from_numpy(result_4d)
    ants.image_write(result_4d_ants, output_dir + 'result_4d.nii')
    JacDet_4d_ants = ants.from_numpy(JacDet_4d)
    ants.image_write(JacDet_4d_ants, output_dir + 'JacDet_4d.nii')
      
    # display Registered image
    #pl.ImagePlot(result_4d, x = 0, y = 2)
    
    
    
def display_jacobian_4d(output_dir, mask_path_close):
    # display Jacobian determinant
    
    # load jacobian
    result_4d_ants = ants.image_read(output_dir + 'result_4d.nii')
    result_4d = result_4d_ants.numpy()
    
    # load jacobian
    JacDet_4d_ants = ants.image_read(output_dir + 'JacDet_4d.nii')
    JacDet_4d = JacDet_4d_ants.numpy()
    
    # load Mask
    ref = 0
    mask_close = ants.image_read(mask_path_close)
    mask_close_np = mask_close.numpy()[:,:,:,ref]
    mask_close_rep = np.tile(mask_close_np[...,np.newaxis], [1,1,1,np.shape(mask_close.numpy())[-1]])
    
    # overlay masked jacobian
    JacDet_masked = np.ma.masked_where(mask_close_rep==0, JacDet_4d)
    pl.ImagePlot(result_4d, x = 0, y = 2,  overlay = JacDet_masked, alpha=0.5)


if __name__ == '__main__':
    
    date = str(sys.argv[1]) # '2020-08-20_vo' 
    pfile = str(sys.argv[2]) # 'P59904'
    #image_path = '/data/larson4/UTE_Lung/2020-01-24_vo/cfl/P37888/MRI_Raw_pr_rec_12bins_v2.nii'
    #image_path = '/data/larson4/UTE_Lung/2020-07-30_vo/cfl/P48128/MRI_Raw_pr_rec_v2.nii'
    #image_path = '/data/larson2/UTE_Lung2/2020-09-21_vo/cfl/P32768/MRI_Raw_pr_rec_v2.nii'
    #image_path = '/data/larson2/UTE_Lung2/2020-11-10_vo/cfl/P12800/MRI_Raw_pr_rec_v2.nii'
    image_path = '/data/larson4/UTE_Lung/'+ date +'/cfl/' + pfile + '/MRI_Raw_pr_rec_v3.nii'
    #image_path = '/data/larson4/UTE_Lung/2020-09-14_vo/cfl/P15872/MRI_Raw_pr_rec_v2.nii'
    
    #output_dir = '/data/larson4/UTE_Lung/2020-01-24_vo/reg/P37888/ants_syn_concat/'
    #output_dir = '/data/larson4/UTE_Lung/2020-07-30_vo/reg/P48128/ants_syn_concat/'
    #output_dir = '/data/larson2/UTE_Lung2/2020-09-21_vo/reg/P32768/ants_syn_concat/'
    #output_dir = '/data/larson2/UTE_Lung2/2020-11-10_vo/reg/P12800/ants_syn_concat/'
    output_dir = '/data/larson4/UTE_Lung/' + date + '/reg/' + pfile + '/ants_syn_concat/'
    #output_dir = '/data/larson4/UTE_Lung/2020-09-14_vo/reg/P15872/ants_syn_concat/'
    
    mask_dir = '/data/larson4/UTE_Lung/'+ date + '/seg/'+ pfile + '/'
    #mask_dir = '/data/larson4/UTE_Lung/2020-09-14_vo/seg/P15872/'
    mask_path_dilate = mask_dir + "lung_mask_dilate.nii"
    mask_path_close = mask_dir + "lung_mask_close.nii"
    
    ants_reg(image_path, output_dir, mask_path_dilate)
    display_jacobian_4d(output_dir, mask_path_close)




