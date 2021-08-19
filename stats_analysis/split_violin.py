#!/usr/bin/env python
# coding: utf-8


import seaborn as sns
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

def split_violin_jacobian(output_dir, mask_path_close, output_dir1, mask_path_close1):
    
    # load jacobiant determiant 1
    JacDet = sitk.ReadImage(output_dir + "JacDet_4d.nii")
    JacDet_np = sitk.GetArrayFromImage(JacDet)
    Jac_shape = np.shape(JacDet_np)    
    
    # load mask 1
    ref = 0
    mask_close = sitk.ReadImage(mask_path_close)
    mask_close_np = sitk.GetArrayFromImage(mask_close)
    mask_close_rep = np.tile(mask_close_np[ref,:,:,:], [Jac_shape[0],1,1,1])

    # create dataframe 
    # col 1: Jacobian determiant
    JacDet_np_col = np.reshape(JacDet_np-1, (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
    df = pd.DataFrame(data=JacDet_np_col, columns=["Jac"])
    # col 2: Mask
    mask_re = np.reshape(mask_close_rep, (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
    df["mask"] = (mask_re != 0)
    # col 3: phase
    tmp = np.linspace(1,Jac_shape[0],Jac_shape[0], dtype = np.uint8)
    phase = np.tile(tmp[:,np.newaxis,np.newaxis,np.newaxis], (1,Jac_shape[1],Jac_shape[2],Jac_shape[3]))
    np.shape(phase)
    phase_re = np.reshape(phase, (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
    df['phase'] = phase_re
    # col 4: scan 1 or 2
    df['scan'] = np.ones(((Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]),1), dtype = np.uint8)
    # select based on mask
    df_new = df[df["mask"]]
    
    
    # load Jacobian determinant 2
    JacDet1 = sitk.ReadImage(output_dir1 + "JacDet_4d.nii")
    JacDet_np1 = sitk.GetArrayFromImage(JacDet1)
    Jac_shape1 = np.shape(JacDet_np1)
    # load mask 
    mask_close1 = sitk.ReadImage(mask_path_close1)
    mask_close_np1 = sitk.GetArrayFromImage(mask_close1)
    mask_close_rep1 = np.tile(mask_close_np1[ref,:,:,:], [Jac_shape1[0],1,1,1])
 
    # create dataframe 2
    # col 1: Jacobian determiant
    JacDet_np_col1 = np.reshape(JacDet_np1-1, (Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]))
    df1 = pd.DataFrame(data=JacDet_np_col1, columns=["Jac"])
    # col 2: Mask
    mask_re1 = np.reshape(mask_close_rep1, (Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]))
    df1["mask"] = (mask_re1 != 0)
    # col 3: phase
    tmp1 = np.linspace(1,Jac_shape[0],Jac_shape[0], dtype = np.uint8)
    phase1 = np.tile(tmp1[:,np.newaxis,np.newaxis,np.newaxis], (1,Jac_shape1[1],Jac_shape1[2],Jac_shape1[3]))
    np.shape(phase)
    phase_re1 = np.reshape(phase1, (Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]))
    df1['phase'] = phase_re1
    # col 4: scan 1 or 2 
    df1['scan'] = 2 * np.ones(((Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]),1), dtype = np.uint8)
    
    # select based on mask
    df_new1 = df1[df1["mask"]]
    
    
    # combine 2 data frames
    df_concat = pd.concat([df_new, df_new1], ignore_index=True, sort=False)
    
    # split violin plot
    plt.figure()
    sns.set_context("talk")
    sns_plot = sns.violinplot(data=df_concat, x="phase", y="Jac", hue="scan",
                   split=True, inner="quart", linewidth=1.5, bw = .2)
    sns_plot.set_ylim(-0.2, 0.6)
    sns_plot.set(ylabel='Regional Ventilation (Jac - 1)')
    sns_plot.figure.savefig(output_dir + 'JD_split_violin.png', bbox_inches='tight', dpi=300)
    
    return df_concat
    
def split_violin_sv(output_dir, mask_path_close, output_dir1, mask_path_close1):
    
    # load registered image 1
    SV_4d = sitk.ReadImage(output_dir + "SV_sm_4d.nii")
    SV = np.abs(sitk.GetArrayFromImage(SV_4d))
    result_shape = np.shape(SV)
    
    # load mask 1
    ref = 0
    mask_close = sitk.ReadImage(mask_path_close)
    mask_close_np = sitk.GetArrayFromImage(mask_close)
    mask_close_rep = np.tile(mask_close_np[ref,:,:,:], [result_shape[0],1,1,1])
        
    # create dataframe 
    # col 1: Specific Ventilation
    SV_np_col = np.reshape(SV, (result_shape[0] * result_shape[1] * result_shape[2] * result_shape[3]))
    df = pd.DataFrame(data=SV_np_col, columns=["SV"])
    # col 2: Mask
    mask_re = np.reshape(mask_close_rep, (result_shape[0] * result_shape[1] * result_shape[2] * result_shape[3]))
    df["mask"] = (mask_re != 0)
    # col 3: phase
    tmp = np.linspace(1,result_shape[0],result_shape[0], dtype = np.uint8)
    phase = np.tile(tmp[:,np.newaxis,np.newaxis,np.newaxis], (1,result_shape[1],result_shape[2],result_shape[3]))
    phase_re = np.reshape(phase, (result_shape[0] * result_shape[1] * result_shape[2] * result_shape[3]))
    df['phase'] = phase_re
    # col 4: scan 1 or 2
    df['scan'] = np.ones(((result_shape[0] * result_shape[1] * result_shape[2] * result_shape[3]),1), dtype = np.uint8)
    # select based on mask
    df_new = df[df["mask"]]
    
    
    # load registered image 2
    SV_4d1 = sitk.ReadImage(output_dir1 + "SV_sm_4d.nii")
    SV1 = np.abs(sitk.GetArrayFromImage(SV_4d1))
    result_shape1 = np.shape(SV1)
    
    # load mask 
    mask_close1 = sitk.ReadImage(mask_path_close1)
    mask_close_np1 = sitk.GetArrayFromImage(mask_close1)
    mask_close_rep1 = np.tile(mask_close_np1[ref,:,:,:], [result_shape1[0],1,1,1])
      
    # create dataframe 2
    # col 1: Specific Ventilation
    SV_np_col1 = np.reshape(SV1, (result_shape1[0] * result_shape1[1] * result_shape1[2] * result_shape1[3]))
    df1 = pd.DataFrame(data=SV_np_col1, columns=["SV"])
    # col 2: Mask
    mask_re1 = np.reshape(mask_close_rep1, (result_shape1[0] * result_shape1[1] * result_shape1[2] * result_shape1[3]))
    df1["mask"] = (mask_re1 != 0)
    # col 3: phase
    tmp1 = np.linspace(1,result_shape1[0],result_shape1[0], dtype = np.uint8)
    phase1 = np.tile(tmp1[:,np.newaxis,np.newaxis,np.newaxis], (1,result_shape1[1],result_shape1[2],result_shape1[3]))
    np.shape(phase)
    phase_re1 = np.reshape(phase1, (result_shape1[0] * result_shape1[1] * result_shape1[2] * result_shape1[3]))
    df1['phase'] = phase_re1
    # col 4: scan 1 or 2 
    df1['scan'] = 2 * np.ones(((result_shape1[0] * result_shape1[1] * result_shape1[2] * result_shape1[3]),1), dtype = np.uint8)
    # select based on mask
    df_new1 = df1[df1["mask"]]
    
    # combine 2 data frames
    df_concat = pd.concat([df_new, df_new1], ignore_index=True, sort=False)
    
    # split violin plot
    plt.figure()
    sns.set_context("talk")
    sns_plot = sns.violinplot(data=df_concat, x="phase", y="SV", hue="scan",
                   split=True, inner="quart", linewidth=1.5, bw = .2)
    sns_plot.set_ylim(-1, 1)
    sns_plot.set(ylabel='Specific Ventilation')
    sns_plot.figure.savefig(output_dir + 'SV_split_violin_k5.png', bbox_inches='tight', dpi=300)
    return df_concat

if __name__ == '__main__':
    # 3DnT, sliding motion, ants
    output_dir_list = ['/data/larson4/UTE_Lung/2020-07-30_vo/reg/P44544/',
                       '/data/larson4/UTE_Lung/2020-08-20_vo/reg/P56320/',
                       '/data/larson4/UTE_Lung/2020-09-14_vo/reg/P12288/',
                       '/data/larson4/UTE_Lung/2020-09-21_vo/reg/P28672/',
                       '/data/larson4/UTE_Lung/2020-11-10_vo/reg/P08704/',
                       '/data/larson4/UTE_Lung/2021-03-12_vo/reg/P86528/']
    
    # load tight mask
    mask_dir_list = ['/data/larson4/UTE_Lung/2020-07-30_vo/seg/P44544/',
                     '/data/larson4/UTE_Lung/2020-08-20_vo/seg/P56320/',
                     '/data/larson4/UTE_Lung/2020-09-14_vo/seg/P12288/',
                     '/data/larson4/UTE_Lung/2020-09-21_vo/seg/P28672/',
                     '/data/larson4/UTE_Lung/2020-11-10_vo/seg/P08704/',
                     '/data/larson4/UTE_Lung/2021-03-12_vo/seg/P86528/']
    
    # 3DnT, sliding motion, ants
    output_dir_list1 = ['/data/larson4/UTE_Lung/2020-07-30_vo/reg/P48128/',
                        '/data/larson4/UTE_Lung/2020-08-20_vo/reg/P59904/',
                        '/data/larson4/UTE_Lung/2020-09-14_vo/reg/P15872/',
                        '/data/larson4/UTE_Lung/2020-09-21_vo/reg/P32768/',
                        '/data/larson4/UTE_Lung/2020-11-10_vo/reg/P12800/',
                        '/data/larson4/UTE_Lung/2021-03-12_vo/reg/P90112/']
    
    # load tight mask
    mask_dir_list1 = ['/data/larson4/UTE_Lung/2020-07-30_vo/seg/P48128/',
                      '/data/larson4/UTE_Lung/2020-08-20_vo/seg/P59904/',
                      '/data/larson4/UTE_Lung/2020-09-14_vo/seg/P15872/',
                      '/data/larson4/UTE_Lung/2020-09-21_vo/seg/P32768/',
                      '/data/larson4/UTE_Lung/2020-11-10_vo/seg/P12800/',
                      '/data/larson4/UTE_Lung/2021-03-12_vo/seg/P90112/']

    df_all_jd = pd.DataFrame()
    df_all_sv = pd.DataFrame()
    reg = 'ants_syn_concat/'
    for ind in range(len(output_dir_list)):        
        output_dir = output_dir_list[ind] + reg
        mask_path_close = mask_dir_list[ind] + "lung_mask_close.nii"
        output_dir1 = output_dir_list1[ind] + reg
        mask_path_close1 = mask_dir_list1[ind] + "lung_mask_close.nii"
        
        df_tmp = split_violin_jacobian(output_dir, mask_path_close, output_dir1, mask_path_close1)
        df_tmp['volunteer'] = (ind+1)
        df_tmp['reg'] = 'JD_ants'  
        df_all_jd = df_all_jd.append(df_tmp, ignore_index=True)
        
        df_tmp = split_violin_sv(output_dir, mask_path_close, output_dir1, mask_path_close1)
        df_tmp['volunteer'] = (ind+1)
        df_tmp['reg'] = 'SV_ants'  
        df_all_sv = df_all_sv.append(df_tmp, ignore_index=True)
 
    reg1 = '3DnT_BSpline/'
    for ind in range(len(output_dir_list)):        
        output_dir = output_dir_list[ind] + reg1
        mask_path_close = mask_dir_list[ind] + "lung_mask_close.nii"
        output_dir1 = output_dir_list1[ind] + reg1
        mask_path_close1 = mask_dir_list1[ind] + "lung_mask_close.nii"
        
        df_tmp = split_violin_jacobian(output_dir, mask_path_close, output_dir1, mask_path_close1)
        df_tmp['volunteer'] = (ind+1)
        df_tmp['reg'] = 'JD_3DnT'  
        df_all_jd = df_all_jd.append(df_tmp, ignore_index=True)
        
        df_tmp = split_violin_sv(output_dir, mask_path_close, output_dir1, mask_path_close1)
        df_tmp['volunteer'] = (ind+1)
        df_tmp['reg'] = 'SV_3DnT'  
        df_all_sv = df_all_sv.append(df_tmp, ignore_index=True)
      
    #df_all['Jac'] = df_all['Jac'].fillna(df_all['SV'])
    sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.close('all')
    g1 = sns.catplot(data=df_all_jd, x="phase", y="Jac", kind = "violin", col="reg", row="volunteer", margin_titles=True, hue="scan", split=True, inner="quart", linewidth=1.5, bw = .2, col_order = ['JD_3DnT','JD_ants'], palette='deep', aspect = 1.2)
    g1.map_dataframe(sns.pointplot, x="phase", y="Jac",ci=None, hue="scan", dodge=0.5, estimator=np.median, palette = "tab10",markers =['.','.'],scale = 0.8)
    g1.set(ylim=(-0.5,0.8), xlabel = 'phase', ylabel = "Regional Ventilation")
    g1.savefig(output_dir + 'split_violin_jd.png', bbox_inches='tight', dpi=300)
    
    plt.close()   
    g2 = sns.catplot(data=df_all_sv, x="phase", y="SV", kind = "violin", col="reg", row="volunteer", margin_titles=True, hue="scan", split=True, inner="quart", linewidth=1.5, bw = .2, col_order = ['SV_3DnT','SV_ants'], palette='deep', aspect = 1.2)
    g2.map_dataframe(sns.pointplot, x="phase", y="SV",ci=None, hue="scan", dodge=0.5, estimator=np.median, palette = "tab10",markers =['.','.'],scale = 0.8)
    g2.set(ylim=(-0.5,0.8), xlabel = 'phase', ylabel = "Specific Ventilation")
    g2.savefig(output_dir + 'split_violin_sv.png', bbox_inches='tight', dpi=300)
    #g1 = sns.catplot(data=df_all, x="phase", y="Jac", kind = "point", row="reg", col="volunteer", margin_titles=True, hue="scan", capsize=.2, dodge=True, ci="sd", row_order = ['JD_3DnT','JD_ants','SV_3DnT','SV_ants' ])
    #g2.savefig(output_dir + 'point_plot_all.png', bbox_inches='tight', dpi=300)
    
    # save median to file
    df_jd_median = df_all_jd.groupby(['reg','phase','volunteer']).median()
    df_jd_median.reset_index(inplace=True)
    df_sv_median = df_all_sv.groupby(['reg','phase','volunteer']).median()
    df_sv_median.reset_index(inplace=True)
    df_median = pd.concat([df_jd_median, df_sv_median], ignore_index=True, sort=False)
    df_median['Jac'] = df_median['Jac'].fillna(df_median['SV'])
    df_median = df_median.rename(columns={"Jac":"ventilation"})
    df_median.to_pickle('/home/ftan1/Downloads/median.pkl')
    