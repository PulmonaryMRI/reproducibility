#!/usr/bin/env python
# coding: utf-8

# SimpleItk env
import seaborn as sns
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


def jacobian_coefficient_of_variation(output_dir, mask_dir, output_dir1):
    # load jacobian determinant of scan 1
    print(output_dir)
    JacDet = sitk.ReadImage(output_dir + "JacDet_4d.nii")
    JacDet_np = sitk.GetArrayFromImage(JacDet)
    Jac_shape = np.shape(JacDet_np)
    
    # load tight mask of scan 1  
    ref = 0
    mask_path_close = mask_dir + "lung_mask_close.nii"
    mask_close = sitk.ReadImage(mask_path_close)
    mask_close_np = sitk.GetArrayFromImage(mask_close)
    mask_close_rep = np.tile(mask_close_np[ref,:,:,:], [Jac_shape[0],1,1,1])   
    
    # create dataframe 1
    # col 1: Jacobian determiant
    JacDet_np_col = np.reshape(np.log(JacDet_np), (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
    df = pd.DataFrame(data=JacDet_np_col, columns=["Jac1_lg"])
    # col 2: Mask
    mask_re = np.reshape(mask_close_rep, (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
    df["mask"] = (mask_re != 0)
    # col 3: phase
    tmp = np.linspace(1,12,12, dtype = np.uint8)
    phase = np.tile(tmp[:,np.newaxis,np.newaxis,np.newaxis], (1,Jac_shape[1],Jac_shape[2],Jac_shape[3]))
    phase_re = np.reshape(phase, (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
    df['phase'] = phase_re
    
    # load jacobian determinant of scan 2
    print(output_dir1)
    JacDet1 = sitk.ReadImage(output_dir1 + "JacDet_4d_tx.nii")
    JacDet_np1 = sitk.GetArrayFromImage(JacDet1)
    
    # col 4: Jacobian determiant 2 tx
    JacDet_np_col1 = np.reshape(np.log(JacDet_np1), (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
    df["Jac2_lg"] = JacDet_np_col1

    # select based on mask
    df_new = df[df["mask"]]
    
    # within subject CV, log method
    df_new['diff'] =  (df_new['Jac1_lg'] - df_new['Jac2_lg']) ** 2 / 2
    df_group = df_new.groupby(['phase']).mean()
    df_group.reset_index(inplace=True)
    CV = np.exp(np.sqrt(df_group['diff'])) - 1
    
    return CV

def sv_coefficient_of_variation(output_dir, mask_dir, output_dir1):
    # load SV of scan 1
    print(output_dir)
    SV_4d = sitk.ReadImage(output_dir + "SV_sm_4d.nii")
    SV = sitk.GetArrayFromImage(SV_4d)
    result_shape = np.shape(SV)
    
    # load tight mask of scan 1
    ref = 0
    mask_path_close = mask_dir + "lung_mask_close.nii"
    mask_close = sitk.ReadImage(mask_path_close)
    mask_close_np = sitk.GetArrayFromImage(mask_close)
    mask_close_rep = np.tile(mask_close_np[ref,:,:,:], [result_shape[0],1,1,1])   
    
    # create dataframe 1
    # col 1: Specific Ventilation
    SV_np_col = np.reshape(SV, (result_shape[0] * result_shape[1] * result_shape[2] * result_shape[3]))
    df = pd.DataFrame(data=np.log(SV_np_col+1), columns=["SV1_lg"])
    # col 2: Mask
    mask_re = np.reshape(mask_close_rep, (result_shape[0] * result_shape[1] * result_shape[2] * result_shape[3]))
    df["mask"] = (mask_re != 0)
    # col 3: phase
    tmp = np.linspace(1,12,12, dtype = np.uint8)
    phase = np.tile(tmp[:,np.newaxis,np.newaxis,np.newaxis], (1,result_shape[1],result_shape[2],result_shape[3]))
    phase_re = np.reshape(phase, (result_shape[0] * result_shape[1] * result_shape[2] * result_shape[3]))
    df['phase'] = phase_re
    
    # load registered image 2
    print(output_dir1)
    SV_4d1 = sitk.ReadImage(output_dir1 + "SV_4d_tx.nii")
    SV1 = sitk.GetArrayFromImage(SV_4d1)
    result_shape1 = np.shape(SV1)
    
    # col 4: Specific Ventilation tx
    SV_np_col1 = np.reshape(SV1, (result_shape1[0] * result_shape1[1] * result_shape1[2] * result_shape1[3]))
    df["SV2_lg"] = np.log(SV_np_col1 + 1)

    # select based on mask
    df_new = df[df["mask"]]
    
    # within subject CV, log method
    df_new['diff'] =  (df_new['SV1_lg'] - df_new['SV2_lg']) ** 2 / 2
    df_group = df_new.groupby(['phase']).mean()
    df_group.reset_index(inplace=True)
    CV = np.exp(np.sqrt(df_group['diff'])) - 1
    
    return CV

if __name__ == '__main__':
    
    # scan 1 image dir list
    output_dir_list1 = ['/data/larson4/UTE_Lung/2020-07-30_vo/reg/P44544/',
                       '/data/larson4/UTE_Lung/2020-08-20_vo/reg/P56320/',
                       '/data/larson4/UTE_Lung/2020-09-14_vo/reg/P12288/',
                       '/data/larson4/UTE_Lung/2020-09-21_vo/reg/P28672/',
                       '/data/larson4/UTE_Lung/2020-11-10_vo/reg/P08704/',
                       '/data/larson4/UTE_Lung/2021-03-12_vo/reg/P86528/']
    
    # scan 1 tight mask list
    mask_dir_list1 = ['/data/larson4/UTE_Lung/2020-07-30_vo/seg/P44544/',
                     '/data/larson4/UTE_Lung/2020-08-20_vo/seg/P56320/',
                     '/data/larson4/UTE_Lung/2020-09-14_vo/seg/P12288/',
                     '/data/larson4/UTE_Lung/2020-09-21_vo/seg/P28672/',
                     '/data/larson4/UTE_Lung/2020-11-10_vo/seg/P08704/',
                     '/data/larson4/UTE_Lung/2021-03-12_vo/seg/P86528/']
    
    # scan 2 image dir list
    output_dir_list2 = ['/data/larson4/UTE_Lung/2020-07-30_vo/reg/P48128/',
                        '/data/larson4/UTE_Lung/2020-08-20_vo/reg/P59904/',
                        '/data/larson4/UTE_Lung/2020-09-14_vo/reg/P15872/',
                        '/data/larson4/UTE_Lung/2020-09-21_vo/reg/P32768/',
                        '/data/larson4/UTE_Lung/2020-11-10_vo/reg/P12800/',
                        '/data/larson4/UTE_Lung/2021-03-12_vo/reg/P90112/']
    
    # scan 2 tight mask list
    mask_dir_list2 = ['/data/larson4/UTE_Lung/2020-07-30_vo/seg/P48128/',
                      '/data/larson4/UTE_Lung/2020-08-20_vo/seg/P59904/',
                      '/data/larson4/UTE_Lung/2020-09-14_vo/seg/P15872/',
                      '/data/larson4/UTE_Lung/2020-09-21_vo/seg/P32768/',
                      '/data/larson4/UTE_Lung/2020-11-10_vo/seg/P12800/',
                      '/data/larson4/UTE_Lung/2021-03-12_vo/seg/P90112/']
    
    # registration method list
    reg_list = ['3DnT_BSpline/','sliding_motion/','ants_syn_concat/']
    
    # initialization
    df_all = pd.DataFrame()
    
    # registration: 3d+t b-spline, ventilation: JD 
    reg = 0
    for ind in range(len(output_dir_list1)):
        output_dir1 = output_dir_list1[ind] + reg_list[reg]
        output_dir2 = output_dir_list2[ind] + reg_list[reg]
        mask_dir = mask_dir_list1[ind]
        
        df_tmp = pd.DataFrame()
        df_tmp['CV'] = jacobian_coefficient_of_variation(output_dir1, mask_dir, output_dir2)
        df_tmp['vol'] = (ind + 1) * np.ones(len(df_tmp['CV']))
        df_tmp['phase'] = range(1, len(df_tmp['CV']) + 1)
        df_tmp['Ventilation, Registration'] = 'Regional, B-Spline'
        # append dataframe
        df_all = df_all.append(df_tmp, ignore_index=True)
    
    # registration: SyN, ventilation: JD 
    reg = 2
    for ind in range(len(output_dir_list1)):
        output_dir1 = output_dir_list1[ind] + reg_list[reg]
        output_dir2 = output_dir_list2[ind] + reg_list[reg]
        mask_dir = mask_dir_list1[ind]
        
        df_tmp = pd.DataFrame()
        df_tmp['CV'] = jacobian_coefficient_of_variation(output_dir1, mask_dir, output_dir2)
        df_tmp['vol'] = (ind + 1) * np.ones(len(df_tmp['CV']))
        df_tmp['phase'] = range(1, len(df_tmp['CV']) + 1)
        df_tmp['Ventilation, Registration'] = 'Regional, SyN'
        # append dataframe
        df_all = df_all.append(df_tmp, ignore_index=True)
    
    # registration: 3d+t b-spline, ventilation: SV
    reg = 0
    for ind in range(len(output_dir_list1)):
        output_dir1 = output_dir_list1[ind] + reg_list[reg]
        output_dir2 = output_dir_list2[ind] + reg_list[reg]
        mask_dir = mask_dir_list1[ind]
        
        df_tmp = pd.DataFrame()
        df_tmp['CV'] = sv_coefficient_of_variation(output_dir1, mask_dir, output_dir2)
        df_tmp['vol'] = (ind + 1) * np.ones(len(df_tmp['CV']))
        df_tmp['phase'] = range(1, len(df_tmp['CV']) + 1)
        df_tmp['Ventilation, Registration'] = 'Specific, B-Spline'
        # append dataframe
        df_all = df_all.append(df_tmp, ignore_index=True)
    
    # registration: SyN, ventilation: SV
    reg = 2
    for ind in range(len(output_dir_list1)):
        output_dir1 = output_dir_list1[ind] + reg_list[reg]
        output_dir2 = output_dir_list2[ind] + reg_list[reg]
        mask_dir = mask_dir_list1[ind]
        
        df_tmp = pd.DataFrame()
        df_tmp['CV'] = sv_coefficient_of_variation(output_dir1, mask_dir, output_dir2)
        df_tmp['vol'] = (ind + 1) * np.ones(len(df_tmp['CV']))
        df_tmp['phase'] = range(1, len(df_tmp['CV']) + 1)
        df_tmp['Ventilation, Registration'] = 'Specific, SyN'
        # append dataframe
        df_all = df_all.append(df_tmp, ignore_index=True)

# plot with seaborn package
sns.set_context("talk", font_scale = 1.2)
g = sns.catplot(data=df_all, x="phase", y="CV", hue="Ventilation, Registration", kind = "box", palette = 'colorblind', legend_out = True, height = 8, aspect = 1.5)
plt.ylabel("coefficient of variation")
plt.xlabel("respiratory phase")
g.savefig(output_dir_list1[-1] + reg_list[0] + 'coefficient_of_variation.png', bbox_inches='tight', dpi = 300)