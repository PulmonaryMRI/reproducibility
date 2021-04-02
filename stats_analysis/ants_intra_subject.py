#!/usr/bin/env python
# coding: utf-8


import ants
import sigpy.plot as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fl


def ants_reg_diff(dir_path1, mask_dir1, dir_path2, mask_dir2, reg):
    
    # direcotry for scan 1
    output_dir1 = dir_path1 + reg
    print(output_dir1)
    # load registered image for scan 1
    scan1_path = output_dir1 + 'result_4d.nii'
    scan1 = ants.image_read(scan1_path)
    #pl.ImagePlot(scan1.numpy(), x = -4, y = -2)
    # load mask for scan 1
    mask_path1 = mask_dir1 + "lung_mask_dilate.nii"
    mask1 = ants.image_read(mask_path1)
    # load jacobian determinant for scan 1
    Jac_dir1 = output_dir1 + 'JacDet_4d.nii'
    JacDet1 = ants.image_read(Jac_dir1)
    
    # directory for scan 2
    output_dir2 = dir_path2 + reg
    print(output_dir2)
    # load registered image for scan 2
    scan2_path = output_dir2 + 'result_4d.nii'
    scan2 = ants.image_read(scan2_path)
    #pl.ImagePlot(scan2.numpy(), x = -4, y = -2)
    # load mask for scan 2
    mask_path2 = mask_dir2 + "lung_mask_dilate.nii"
    mask2 = ants.image_read(mask_path2)
    # load jacobian determianant for scan2
    Jac_dir2 = output_dir2 + 'JacDet_4d.nii'
    JacDet2 = ants.image_read(Jac_dir2)
    
    # initialize
    scan2_tx_np = np.zeros(np.shape(scan2))
    JacDet2_tx_np = np.zeros(np.shape(JacDet1))
    
    # register phase by phase, 4d registration not working for ants
    for phase in range(np.shape(scan1)[-1]):
        # set fixed, moving and mask        
        fi = ants.from_numpy(scan1[:,:,:,phase])
        mv = ants.from_numpy(scan2[:,:,:,phase])
        ma = ants.from_numpy((mask1[:,:,:,phase]+mask2[:,:,:,phase]))
        # SyN registration
        mytx = ants.registration(fixed = fi, moving = mv, type_of_transform='SyN', reg_iterations=(100,70,50,20,0), mask=ma)
        #pl.ImagePlot(mytx['warpedmovout'].numpy(), x = -3, y = -1,  overlay = fi, alpha=0.5)
        scan2_tx_np[:,:,:,phase] = mytx['warpedmovout'].numpy()
        # apply transform to Jacobian determiant
        jd_fi = ants.from_numpy(JacDet1[:,:,:,phase])
        jd_mv = ants.from_numpy(JacDet2[:,:,:,phase])
        jd_tx = ants.apply_transforms(fixed = jd_fi, moving = jd_mv, transformlist = mytx['fwdtransforms'])
        JacDet2_tx_np[:,:,:,phase] = jd_tx.numpy()
    # save the 4D registered image      
    scan2_tx = ants.from_numpy(scan2_tx_np)
    ants.image_write(scan2_tx, output_dir2 + 'result_4d_tx.nii')
    # save the 4D jacobian determinant
    JacDet2_tx = ants.from_numpy(JacDet2_tx_np)
    ants.image_write(JacDet2_tx, output_dir2 + 'JacDet_4d_tx.nii')


def display_jacobian_diff_4d(dir_path1, mask_dir1, dir_path2, reg):

    # direcotry for scan 1
    output_dir1 = dir_path1 + reg
    print(output_dir1)
    # load registered image for scan 1
    scan1_path = output_dir1 + 'result_4d.nii'
    scan1 = ants.image_read(scan1_path)
    
    # load mask for scan 1
    ref = 0
    mask_path_close = mask_dir1 + "lung_mask_close.nii"
    mask_close = ants.image_read(mask_path_close)
    mask_close_np = mask_close.numpy()[:,:,:,ref]
    mask_close_rep = np.tile(mask_close_np[...,np.newaxis], [1,1,1,np.shape(mask_close.numpy())[-1]])
    
    # load jacobian determinant for scan 1
    Jac_dir1 = output_dir1 + 'JacDet_4d.nii'
    JacDet1 = ants.image_read(Jac_dir1)
    
    output_dir2 = dir_path2 + reg
    
    # load registered jacobian determinant for scan 2
    Jac_tx_dir2 = output_dir2 + 'JacDet_4d_tx.nii'
    JacDet2_tx = ants.image_read(Jac_tx_dir2)
    
    # overlay masked jacobian
    JacDet_diff = np.ma.masked_where(mask_close_rep==0, JacDet1.numpy() - JacDet2_tx.numpy())
    pl.ImagePlot(scan1.numpy(), x = 0, y = 2,  overlay = JacDet_diff, alpha=0.5)

def display_jacobian_diff_2d(dir_path1, mask_dir1, dir_path2, reg):
    # direcotry for scan 1
    output_dir1 = dir_path1 + reg
    print(output_dir1)
    # load registered image for scan 1
    scan1_path = output_dir1 + 'result_4d.nii'
    scan1 = ants.image_read(scan1_path)
    
    # load mask for scan 1
    ref = 0
    mask_path_close = mask_dir1 + "lung_mask_close.nii"
    mask_close = ants.image_read(mask_path_close)
    mask_close_np = mask_close.numpy()[:,:,:,ref]
    mask_close_rep = np.tile(mask_close_np[...,np.newaxis], [1,1,1,np.shape(mask_close.numpy())[-1]])
    
    # load jacobian determinant for scan 1
    Jac_dir1 = output_dir1 + 'JacDet_4d.nii'
    JacDet1 = ants.image_read(Jac_dir1)
    
    output_dir2 = dir_path2 + reg
    
    # load registered jacobian determinant for scan 2
    Jac_tx_dir2 = output_dir2 + 'JacDet_4d_tx.nii'
    JacDet2_tx = ants.image_read(Jac_tx_dir2)
    # overlay masked jacobian
    JacDet_diff = np.ma.masked_where(mask_close_rep==0, JacDet1.numpy() - JacDet2_tx.numpy())
    pl.ImagePlot(scan1.numpy(), x = 0, y = 2,  overlay = JacDet_diff, alpha=0.5)
    
    for phase in range(np.shape(JacDet1)[-1]):
        # display overlaid Jacobian
        fig = plt.figure()
        slc = 64
        # plot grayscale image
        fig = plt.imshow(scan1[:,slc,:,phase].T, cmap = 'gray')
        fig.set_clim(0.0, 0.7 * np.amax(scan1[:,slc,:,phase]))
        # overlay masked jacobian
        fig = plt.imshow(JacDet_diff[:,slc,:,phase].T, cmap = 'coolwarm', alpha = 0.5, vmin = -0.2, vmax = 0.2)
        plt.axis('off')
        plt.colorbar(fig, extend = 'both', label='Regional Ventilation \n Difference[ml/ml]', shrink = 0.8)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.show()
    
        plt.savefig(output_dir1 + 'JD_diff_'+reg[:-1]+'_p' + str(phase) + '.png', transparent=True)
        plt.close()
        
def display_sv_diff_4d(dir_path1, mask_dir1, dir_path2, reg):

    # direcotry for scan 1
    output_dir1 = dir_path1 + reg
    print(output_dir1)
    # load registered image for scan 1
    scan1_path = output_dir1 + 'result_4d.nii'
    scan1 = ants.image_read(scan1_path)
    
    # load mask for scan 1
    ref = 0
    mask_path_close = mask_dir1 + "lung_mask_close.nii"
    mask_close = ants.image_read(mask_path_close)
    mask_close_np = mask_close.numpy()[:,:,:,ref]
    mask_close_rep = np.tile(mask_close_np[...,np.newaxis], [1,1,1,np.shape(mask_close.numpy())[-1]])
    
    # calculate specific ventilation for scan 1
    ## gaussian filtering, taking lung border into account
    density1 = fl.uniform_filter(mask_close_rep, size = (5,5,5,0))
    result_gauss1 = fl.gaussian_filter(scan1.numpy() * mask_close_rep, (2,2,2,0), truncate = 1) # gaussian kernel width = 3 
    result_gauss_dens1 = result_gauss1 / (density1 + np.finfo(float).eps)
    ## calculate specific ventilation
    result_rep1 = np.tile(result_gauss_dens1[:,:,:,ref][:,:,:,np.newaxis], [1,1,1, np.shape(scan1.numpy())[-1]])
    SV1 = (result_rep1 - result_gauss_dens1) / (result_gauss_dens1 + np.finfo(float).eps)
    ## eliminate extreme outliers
    SV1[(SV1<-2) | (SV1>2)] = np.NaN
    
    output_dir2 = dir_path2 + reg
    
    # load registered jacobian determinant for scan 2
    scan2_tx_path = output_dir2 + 'result_4d_tx.nii'
    scan2_tx = ants.image_read(scan2_tx_path)
    
    # calculate specific ventilation for registered scan 2 
    ## gaussian filtering, taking lung border into account
    density2 = fl.uniform_filter(mask_close_rep, size = (5,5,5,0))
    result_gauss2 = fl.gaussian_filter(scan2_tx.numpy() * mask_close_rep, (2,2,2,0), truncate = 1) # gaussian kernel width = 3 
    result_gauss_dens2 = result_gauss2 / (density2 + np.finfo(float).eps)
    ## calculate specific ventilation
    result_rep2 = np.tile(result_gauss_dens2[:,:,:,ref][:,:,:,np.newaxis], [1,1,1, np.shape(scan1.numpy())[-1]])
    SV2_tx = (result_rep2 - result_gauss_dens2) / (result_gauss_dens2 + np.finfo(float).eps)
    ## eliminate extreme outliers
    SV2_tx[(SV2_tx<-2) | (SV2_tx>2)] = np.NaN
    
    # overlay masked jacobian
    SV_diff = np.ma.masked_where(mask_close_rep==0, SV1 - SV2_tx)
    pl.ImagePlot(scan1.numpy(), x = 0, y = 2,  overlay = SV_diff, alpha=0.5)

def display_sv_diff_2d(dir_path1, mask_dir1, dir_path2, reg):
    # direcotry for scan 1
    output_dir1 = dir_path1 + reg
    print(output_dir1)
    # load registered image for scan 1
    scan1_path = output_dir1 + 'result_4d.nii'
    scan1 = ants.image_read(scan1_path)
    
    # load mask for scan 1
    ref = 0
    mask_path_close = mask_dir1 + "lung_mask_close.nii"
    mask_close = ants.image_read(mask_path_close)
    mask_close_np = mask_close.numpy()[:,:,:,ref]
    mask_close_rep = np.tile(mask_close_np[...,np.newaxis], [1,1,1,np.shape(mask_close.numpy())[-1]])
    
    # calculate specific ventilation for scan 1
    ## gaussian filtering, taking lung border into account
    density1 = fl.uniform_filter(mask_close_rep, size = (5,5,5,0))
    result_gauss1 = fl.gaussian_filter(scan1.numpy() * mask_close_rep, (2,2,2,0), truncate = 1) # gaussian kernel width = 3 
    result_gauss_dens1 = result_gauss1 / (density1 + np.finfo(float).eps)
    ## calculate specific ventilation
    result_rep1 = np.tile(result_gauss_dens1[:,:,:,ref][:,:,:,np.newaxis], [1,1,1, np.shape(scan1.numpy())[-1]])
    SV1 = (result_rep1 - result_gauss_dens1) / (result_gauss_dens1 + np.finfo(float).eps)
    ## eliminate extreme outliers
    SV1[(SV1<-2) | (SV1>2)] = np.NaN
    
    output_dir2 = dir_path2 + reg
    
    # load registered jacobian determinant for scan 2
    scan2_tx_path = output_dir2 + 'result_4d_tx.nii'
    scan2_tx = ants.image_read(scan2_tx_path)
    
    # calculate specific ventilation for registered scan 2 
    ## gaussian filtering, taking lung border into account
    density2 = fl.uniform_filter(mask_close_rep, size = (5,5,5,0))
    result_gauss2 = fl.gaussian_filter(scan2_tx.numpy() * mask_close_rep, (2,2,2,0), truncate = 1) # gaussian kernel width = 3 
    result_gauss_dens2 = result_gauss2 / (density2 + np.finfo(float).eps)
    ## calculate specific ventilation
    result_rep2 = np.tile(result_gauss_dens2[:,:,:,ref][:,:,:,np.newaxis], [1,1,1, np.shape(scan1.numpy())[-1]])
    SV2_tx = (result_rep2 - result_gauss_dens2) / (result_gauss_dens2 + np.finfo(float).eps)
    ## eliminate extreme outliers
    SV2_tx[(SV2_tx<-2) | (SV2_tx>2)] = np.NaN
    
    # overlay masked jacobian
    SV_diff = np.ma.masked_where(mask_close_rep==0, SV1 - SV2_tx)
    #pl.ImagePlot(scan1.numpy(), x = 0, y = 2,  overlay = SV_diff, alpha=0.5)
    # direcotry for scan 1
    output_dir1 = dir_path1 + reg
    print(output_dir1)
    # load registered image for scan 1
    scan1_path = output_dir1 + 'result_4d.nii'
    scan1 = ants.image_read(scan1_path)
    
    
    for phase in range(np.shape(scan1)[-1]):
        # display overlaid Jacobian
        fig = plt.figure()
        slc = 64
        # plot grayscale image
        fig = plt.imshow(scan1[:,slc,:,phase].T, cmap = 'gray')
        fig.set_clim(0.0, 0.7 * np.amax(scan1[:,slc,:,phase]))
        # overlay masked jacobian
        fig = plt.imshow(SV_diff[:,slc,:,phase].T, cmap = 'coolwarm', alpha = 0.5, vmin = -0.2, vmax = 0.2)
        plt.axis('off')
        plt.colorbar(fig, extend = 'both', label='Specific Ventilation Difference[ml/ml]', shrink = 0.8)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.show()
    
        plt.savefig(output_dir1 + 'SV_diff_'+reg[:-1]+'_p' + str(phase) + '.png', transparent=True)
        plt.close()
        
if __name__ =='__main__':
        # 3DnT, sliding motion, ants
    output_dir_list1 = ['/data/larson4/UTE_Lung/2020-07-30_vo/reg/P44544/',
                       '/data/larson4/UTE_Lung/2020-08-20_vo/reg/P56320/',
                       '/data/larson4/UTE_Lung/2020-09-14_vo/reg/P12288/',
                       '/data/larson4/UTE_Lung/2020-09-21_vo/reg/P28672/',
                       '/data/larson4/UTE_Lung/2020-11-10_vo/reg/P08704/',
                       '/data/larson4/UTE_Lung/2021-03-12_vo/reg/P86528/']
    
    # load tight mask
    mask_dir_list1 = ['/data/larson4/UTE_Lung/2020-07-30_vo/seg/P44544/',
                     '/data/larson4/UTE_Lung/2020-08-20_vo/seg/P56320/',
                     '/data/larson4/UTE_Lung/2020-09-14_vo/seg/P12288/',
                     '/data/larson4/UTE_Lung/2020-09-21_vo/seg/P28672/',
                     '/data/larson4/UTE_Lung/2020-11-10_vo/seg/P08704/',
                     '/data/larson4/UTE_Lung/2021-03-12_vo/seg/P86528/']
    
    # 3DnT, sliding motion, ants
    output_dir_list2 = ['/data/larson4/UTE_Lung/2020-07-30_vo/reg/P48128/',
                        '/data/larson4/UTE_Lung/2020-08-20_vo/reg/P59904/',
                        '/data/larson4/UTE_Lung/2020-09-14_vo/reg/P15872/',
                        '/data/larson4/UTE_Lung/2020-09-21_vo/reg/P32768/',
                        '/data/larson4/UTE_Lung/2020-11-10_vo/reg/P12800/',
                        '/data/larson4/UTE_Lung/2021-03-12_vo/reg/P90112/']
    
    # load tight mask
    mask_dir_list2 = ['/data/larson4/UTE_Lung/2020-07-30_vo/seg/P48128/',
                      '/data/larson4/UTE_Lung/2020-08-20_vo/seg/P59904/',
                      '/data/larson4/UTE_Lung/2020-09-14_vo/seg/P15872/',
                      '/data/larson4/UTE_Lung/2020-09-21_vo/seg/P32768/',
                      '/data/larson4/UTE_Lung/2020-11-10_vo/seg/P12800/',
                      '/data/larson4/UTE_Lung/2021-03-12_vo/seg/P90112/']

    
    reg_list = ['3DnT_BSpline/','sliding_motion/','ants_syn_concat/']
    
    reg_ind = 0
    for scn in range(len(output_dir_list1)):
        ants_reg_diff(output_dir_list1[scn], mask_dir_list1[scn], output_dir_list2[scn], mask_dir_list2[scn], reg_list[reg_ind])
        display_jacobian_diff_4d(output_dir_list1[scn], mask_dir_list1[scn], output_dir_list2[scn], reg_list[reg_ind])
        display_sv_diff_4d(output_dir_list1[scn], mask_dir_list1[scn], output_dir_list2[scn], reg_list[reg_ind])
        display_jacobian_diff_2d(output_dir_list1[scn], mask_dir_list1[scn], output_dir_list2[scn], reg_list[reg_ind])
        display_sv_diff_2d(output_dir_list1[scn], mask_dir_list1[scn], output_dir_list2[scn], reg_list[reg_ind])
        
    reg_ind = 2
    for scn in range(len(output_dir_list1)):
        ants_reg_diff(output_dir_list1[scn], mask_dir_list1[scn], output_dir_list2[scn], mask_dir_list2[scn], reg_list[reg_ind])
        display_jacobian_diff_4d(output_dir_list1[scn], mask_dir_list1[scn], output_dir_list2[scn], reg_list[reg_ind])
        display_sv_diff_4d(output_dir_list1[scn], mask_dir_list1[scn], output_dir_list2[scn], reg_list[reg_ind])
        display_jacobian_diff_2d(output_dir_list1[scn], mask_dir_list1[scn], output_dir_list2[scn], reg_list[reg_ind])
        display_sv_diff_2d(output_dir_list1[scn], mask_dir_list1[scn], output_dir_list2[scn], reg_list[reg_ind])