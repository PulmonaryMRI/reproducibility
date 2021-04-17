#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:51:52 2021

@author: ftan1
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fl
import pandas as pd
import seaborn as sns
from scipy.stats.stats import pearsonr

def lung_volume(output_dir1, mask_path_close1):
    
    # load jacobiant determiant 1
    JacDet1 = sitk.ReadImage(output_dir1 + "JacDet_4d.nii")
    JacDet_np1 = sitk.GetArrayFromImage(JacDet1)
    Jac_shape1 = np.shape(JacDet_np1)    
    
    # load registered image 1
    result_4d1 = sitk.ReadImage(output_dir1 + "result_4d.nii")
    result_np1 = np.abs(sitk.GetArrayFromImage(result_4d1))
    result_shape1 = np.shape(result_np1)
    
    # load mask 1
    ref = 0
    mask_close1 = sitk.ReadImage(mask_path_close1)
    mask_close_np1 = sitk.GetArrayFromImage(mask_close1)
    mask_close_rep1 = np.tile(mask_close_np1[ref,:,:,:], [Jac_shape1[0],1,1,1])
    
    # calculate mask and vent volume
    mask_volume1 = np.sum(np.reshape(mask_close_np1-mask_close_rep1,(Jac_shape1[0], Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3])), axis=1) * (0.025 ** 3)
    JD_volume1 = np.sum(np.reshape(mask_close_rep1*(JacDet_np1-1),(Jac_shape1[0], Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3])), axis=1) * (0.025 ** 3)
    
    # calculate vent volume
    # Specific Ventilation
    ## gaussian filtering, taking lung border into account
    density1 = fl.uniform_filter(mask_close_rep1, size = (0,5,5,5))
    result_gauss1 = fl.gaussian_filter(result_np1 * mask_close_np1, (0,2,2,2), truncate = 1) # gaussian kernel width = 3 
    result_gauss_dens1 = result_gauss1 / (density1 + np.finfo(float).eps)
    ## calculate specific ventilation
    result_rep1 = np.tile(result_gauss_dens1[ref,:,:,:], [result_shape1[0],1,1,1])
    SV1 = (result_rep1 - result_gauss_dens1) / (result_gauss_dens1 + np.finfo(float).eps)
    ## eliminate extreme outliers
    SV1[(SV1<-1)] = -1
    SV1[(SV1>2)] = 2
    SV_volume1 = np.sum(np.reshape(mask_close_rep1*(SV1), (result_shape1[0], result_shape1[1] * result_shape1[2] * result_shape1[3])), axis=1) * (0.025 ** 3)
    
    
    df = pd.DataFrame()
    
    df['mask'] = mask_volume1
    df['regional'] = JD_volume1
    df['specific'] = SV_volume1
    
    #df_m = df.melt('mask', var_name='method', value_name='volume')  
    
    df['phase'] = np.linspace(1, Jac_shape1[0],Jac_shape1[0])
    df = df.melt(id_vars = ('phase', 'mask'), var_name='method', value_name='volume')  
    return df

def draw_corr(data, **kws):
    r,p = pearsonr(data['mask'], data['volume'])
    ax = plt.gca()
    ax.plot([0,0.5], [0,0.5],'k--')
    ax.text(0.6, data['volume'][data['mask'].idxmax()], 'r = %.2f' % r, fontsize=14)
    
    
if __name__ == '__main__':
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
    
    df_all = pd.DataFrame()
    
    for ind in range(len(output_dir_list1)):        
        output_dir1 = output_dir_list1[ind] + reg_list[0]
        mask_path_close1 = mask_dir_list1[ind] + "lung_mask_close.nii"
        output_dir2 = output_dir_list2[ind] + reg_list[0]
        mask_path_close2 = mask_dir_list2[ind] + "lung_mask_close.nii"
        
        df_1 = lung_volume(output_dir1, mask_path_close1)
        df_2 = lung_volume(output_dir2, mask_path_close2)
        
        df_1['volunteer'] = ind+1
        df_1['registration'] = 'b-spline'
        df_1['scan'] = 1
        df_all = df_all.append(df_1, ignore_index=True)
        
        df_2['volunteer'] = ind+1
        df_2['registration'] = 'b-spline'
        df_2['scan'] = 2
        df_all = df_all.append(df_2, ignore_index=True)
        
    for ind in range(len(output_dir_list1)):        
        output_dir1 = output_dir_list1[ind] + reg_list[2]
        mask_path_close1 = mask_dir_list1[ind] + "lung_mask_close.nii"
        output_dir2 = output_dir_list2[ind] + reg_list[2]
        mask_path_close2 = mask_dir_list2[ind] + "lung_mask_close.nii"
        
        df_1 = lung_volume(output_dir1, mask_path_close1)
        df_2 = lung_volume(output_dir2, mask_path_close2)
        
        df_1['volunteer'] = ind+1
        df_1['registration'] = 'SyN'
        df_1['scan'] = 1
        df_all = df_all.append(df_1, ignore_index=True)
        
        df_2['volunteer'] = ind+1
        df_2['registration'] = 'SyN'
        df_2['scan'] = 2
        df_all = df_all.append(df_2, ignore_index=True)
    
    df_all['ventilation, registration'] = df_all['method'] + ', ' + df_all['registration']
    sns.set_context("talk", font_scale=2, rc={"lines.linewidth": 4})
    #g = sns.relplot(data = df_all, x = "phase", y ="volume", hue="method", style="registration", row = "volunteer", col="scan", palette="deep", ci=None, markers=True, kind = "line", aspect = 1.2)
    g = sns.relplot(data = df_all[df_all["ventilation, registration"]=="regional, b-spline"], x = "phase", y ="volume", row = "volunteer", hue="scan", palette="deep", ci=None, markers=True, kind = "line", aspect = 1.2)
    g.set(xticks = [1,2,3,4,5,6,7,8,9,10,11,12])
    g.savefig(output_dir1 + 'lung_volume.png', bbox_inches='tight', dpi = 300)
    
    g1 = sns.lmplot(data = df_all, x = "mask", y ="volume", hue="ventilation, registration", row = "volunteer", palette="deep", aspect = 1.5, markers='x', hue_order=("regional, b-spline", "regional, SyN","specific, b-spline","specific, SyN"))
    g1.map_dataframe(draw_corr)
    g1.set(xlabel = 'segmentation volume', ylabel = "ventilation volume")
    g1.savefig(output_dir1 + 'lung_volume_corr.png', bbox_inches='tight', dpi = 300)