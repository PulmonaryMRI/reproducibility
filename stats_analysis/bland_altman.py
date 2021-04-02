#!/usr/bin/env python
# coding: utf-8

# SimpleItk env
import seaborn as sns
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fl
from scipy import stats

def bland_altman_jacobian(dir_list, mask_list, dir_list1, mask_list1, reg):
    """
    Cross subject Bland-Altman plot of jacobian determinant

    Parameters
    ----------
    dir_list : list of string
        list of registered image paths scan 1.
    mask_list : list of string
        list of mask paths scan 1.
    dir_list1 : list of string
        list of registered image paths scan 2.
    mask_list1 : list of string
        list of mask paths scan 2.
    reg : string
        registration type. 

    Returns
    -------
    None.

    """
    # Step 1: import data from all scan 1
    # initialize data frame
    df_all = pd.DataFrame(columns = ['Jac','mask','phase','scan','volunteer'])
    
    # loop over all images/jacobian determinants
    for ind in range(len(dir_list)):    
        
        # load jacobian determinant
        output_dir = dir_list[ind] + reg
        print(output_dir)
        JacDet = sitk.ReadImage(output_dir + "JacDet_4d.nii")
        JacDet_np = sitk.GetArrayFromImage(JacDet)
        Jac_shape = np.shape(JacDet_np)
        
        # load tight mask
        mask_dir = mask_list[ind]
        ref = 0
        mask_path_close = mask_dir + "lung_mask_close.nii"
        mask_close = sitk.ReadImage(mask_path_close)
        mask_close_np = sitk.GetArrayFromImage(mask_close)
        mask_close_rep = np.tile(mask_close_np[ref,:,:,:], [Jac_shape[0],1,1,1])
            
        # create dataframe 
        # col 1: Jacobian determiant
        JacDet_np_col = np.reshape(JacDet_np - 1, (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
        df = pd.DataFrame(data=JacDet_np_col, columns=["Jac"])
        # col 2: Mask
        mask_re = np.reshape(mask_close_rep, (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
        df["mask"] = (mask_re != 0)
        # col 3: phase
        tmp = np.linspace(1,Jac_shape[0],Jac_shape[0], dtype = np.uint8)
        phase = np.tile(tmp[:,np.newaxis,np.newaxis,np.newaxis], (1,Jac_shape[1],Jac_shape[2],Jac_shape[3]))
        phase_re = np.reshape(phase, (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
        df['phase'] = phase_re
        # col 4: scan 1 or 2
        df['scan'] = np.ones(((Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]),1), dtype = np.uint8)
        # col 5: volunteer
        df['volunteer'] = (ind+1) * np.ones(((Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]),1), dtype = np.uint8)
    
        # append dataframe
        df_all = df_all.append(df, ignore_index=True)
    
    # select based on mask
    df_new = df_all[df_all["mask"]]
    
    # Step 2: import data from all scan 2's

    # initialize data frame
    df_all1 = pd.DataFrame(columns = ['Jac','mask','phase','scan','volunteer'])
    
    # loop over all subjects
    for ind1 in range(len(dir_list1)):
        # load jacobian determinant
        output_dir1 = dir_list1[ind1] + reg
        print(output_dir1)
        JacDet1 = sitk.ReadImage(output_dir1 + "JacDet_4d.nii")
        JacDet_np1 = sitk.GetArrayFromImage(JacDet1)
        Jac_shape1 = np.shape(JacDet_np1)
    
        # load tight mask
        mask_dir1 = mask_list1[ind1]
        ref1 = 0
        mask_path_close1 = mask_dir1 + "lung_mask_close.nii"
        mask_close1 = sitk.ReadImage(mask_path_close1)
        mask_close_np1 = sitk.GetArrayFromImage(mask_close1)
        mask_close_rep1 = np.tile(mask_close_np1[ref1,:,:,:], [Jac_shape1[0],1,1,1])   
        
        # create dataframe 
        # col 1: Jacobian determiant
        JacDet_np_col1 = np.reshape(JacDet_np1 - 1, (Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]))
        df1 = pd.DataFrame(data=JacDet_np_col1, columns=["Jac"])
        # col 2: Mask
        mask_re1 = np.reshape(mask_close_rep1, (Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]))
        df1["mask"] = (mask_re1 != 0)
        # col 3: phase
        tmp1 = np.linspace(1,Jac_shape1[0],Jac_shape1[0], dtype = np.uint8)
        phase1 = np.tile(tmp1[:,np.newaxis,np.newaxis,np.newaxis], (1,Jac_shape1[1],Jac_shape1[2],Jac_shape1[3]))
        phase_re1 = np.reshape(phase1, (Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]))
        df1['phase'] = phase_re1
        # col 4: scan 1 or 2
        df1['scan'] = np.ones(((Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]),1), dtype = np.uint8)
        # col 5: volunteer
        df1['volunteer'] = (ind1 + 1) * np.ones(((Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]),1), dtype = np.uint8)
    
        # append dataframe
        df_all1 = df_all1.append(df1, ignore_index=True)
    
    # select based on mask
    df_new1 = df_all1[df_all1["mask"]]
    
    
    # Step 3: combine data frames
    
    # calculate mean regional ventilation(jacobian-1) of different phase and volunteer
    df_group = df_new.groupby(['volunteer','phase']).mean()
    df_group.reset_index(inplace=True)
    
    df_group1 = df_new1.groupby(['volunteer','phase']).mean()
    df_group1.reset_index(inplace=True)
    
    # calculate difference and average of mean regional ventilation
    df_group['diff'] = df_group['Jac'] - df_group1['Jac']
    df_group['avg'] = (df_group['Jac'] + df_group1['Jac']) / 2
    
    # mean and std of difference
    std = 1.96 * np.std(df_group["diff"])
    mean = np.mean(df_group["diff"])
    
    # Bland-Altman plot
    sns.set_context("talk", font_scale = 0.85)
    sns.set_style("ticks")
    sns_plot = sns.relplot(data = df_group, x = "avg", y = "diff", hue = "volunteer", style = "phase", palette = "muted", aspect = 1.2)
    sns_plot.set(xlabel='Average of Total Ventilation of Two Scans ', ylabel='Difference of Total Ventilation between Two Scans', xlim=(-0.1,0.2), ylim = (-0.125, 0.125))
    xlim = [-0.1, 0.2]
    # mean and 1.96 std lines
    plt.plot(xlim, [mean+std, mean+std],'k--', xlim, [mean-std, mean-std],'k--')
    plt.plot(xlim, [mean, mean])
    # add text to the lines
    plt.text(0.2, mean, 'Mean \n %.2e' % mean, fontsize=14, horizontalalignment='right', verticalalignment='center', multialignment='right')
    plt.text(0.2, mean+std, '+1.96 SD \n %.2e' % (mean+std), fontsize=14, horizontalalignment='right', verticalalignment='center', multialignment='right')
    plt.text(0.2, mean-std, '-1.96 SD \n %.2e' % (mean-std), fontsize=14, horizontalalignment='right', verticalalignment='center', multialignment='right')
    # save figure
    sns_plot.savefig(output_dir + 'JD_BlandAltman_'+reg[:-1]+'.png', bbox_inches='tight', dpi = 300)
    
    
    # linear regression plot
    df_group['Jac_scan2'] = df_group1['Jac']
    # calculate r    
    slope, intercept, r, p, se = stats.linregress(df_group['Jac'], df_group1['Jac'])
    # linear regression line plot
    sns.set_context("talk", font_scale = 0.85)
    g = sns.relplot(data = df_group, x = "Jac", y = "Jac_scan2", hue = "volunteer", style = "phase", palette = "muted", aspect = 1.2)
    sns.regplot(x=df_group['Jac'], y=df_group1['Jac'], scatter = False)
    g.set(xlim=(-0.05, 0.2), ylim=(-0.05, 0.2),xlabel = 'Total Ventilation of Scan 1', ylabel = 'Total Ventilation of Scan 2')
    g.fig.text(0.7,0.85,'R$^2$={0:.2f}, p={1:.2e} \n TV2 = {2:.2f} + {3:.2f}TV1'.format(r,p,intercept,slope), horizontalalignment='right')
    plt.plot([-0.05, 0.2], [-0.05, 0.2], 'k--')
    g.savefig(output_dir + 'JD_LinearRegression_'+reg[:-1]+'.png', bbox_inches='tight', dpi = 300)


def bland_altman_sv(dir_list, mask_list, dir_list1, mask_list1, reg):
    """
    Cross subject Bland-Altman plot of specific ventilation

    Parameters
    ----------
    dir_list : list of string
        list of registered image paths scan 1.
    mask_list : list of string
        list of mask paths scan 1.
    dir_list1 : list of string
        list of registered image paths scan 2.
    mask_list1 : list of string
        list of mask paths scan 2.
    reg : string
        registration type. 

    Returns
    -------
    None.

    """
    # Step 1: import data from all scan 1
    
    # initialize data frame
    df_all = pd.DataFrame(columns = ['Jac','mask','phase','scan','volunteer'])
    
    # loop over all images/jacobian determinants
    for ind in range(len(dir_list)):    
        
        # output directory
        output_dir = dir_list[ind] + reg
        print(output_dir)
        # load registered image 1
        result_4d = sitk.ReadImage(output_dir + "result_4d.nii")
        result_np = np.abs(sitk.GetArrayFromImage(result_4d))
        result_shape = np.shape(result_np)
        
        # load tight mask
        mask_dir = mask_list[ind]
        ref = 0
        mask_path_close = mask_dir + "lung_mask_close.nii"
        mask_close = sitk.ReadImage(mask_path_close)
        mask_close_np = sitk.GetArrayFromImage(mask_close)
        mask_close_rep = np.tile(mask_close_np[ref,:,:,:], [result_shape[0],1,1,1])
            
        # create dataframe 
        # col 1: Specific Ventilation
        ## gaussian filtering, taking lung border into account
        density = fl.uniform_filter(mask_close_rep, size = (0,5,5,5))
        result_gauss = fl.gaussian_filter(result_np * mask_close_np, (0,2,2,2), truncate = 1) # gaussian kernel width = 3
        result_gauss_dens = result_gauss / (density + np.finfo(float).eps)
        ## calculate specific ventilation
        result_rep = np.tile(result_gauss_dens[ref,:,:,:], [result_shape[0],1,1,1])
        SV = (result_rep - result_gauss_dens) / (result_gauss_dens + np.finfo(float).eps)
        ## eliminate extreme outliers
        SV[(SV<-2) | (SV>2)] = np.NaN
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
        # col 5: volunteer
        df['volunteer'] = (ind+1) * np.ones(((result_shape[0] * result_shape[1] * result_shape[2] * result_shape[3]),1), dtype = np.uint8)
    
        # append dataframe
        df_all = df_all.append(df, ignore_index=True)
    
    # select based on mask
    df_new = df_all[df_all["mask"]]
    
    # Step 2: import data from all scan 2's
    
    # initialize data frame
    df_all1 = pd.DataFrame(columns = ['Jac','mask','phase','scan','volunteer'])
    
    # loop over all subjects
    for ind1 in range(len(dir_list1)):
        # load jacobian determinant
        output_dir1 = dir_list1[ind1] + reg_list[0]
        print(output_dir1)
        # load registered image 2
        result_4d1 = sitk.ReadImage(output_dir1 + "result_4d.nii")
        result_np1 = np.abs(sitk.GetArrayFromImage(result_4d1))
        result_shape1 = np.shape(result_np1)
    
        # load tight mask
        mask_dir1 = mask_list1[ind1]
        ref1 = 0
        mask_path_close1 = mask_dir1 + "lung_mask_close.nii"
        mask_close1 = sitk.ReadImage(mask_path_close1)
        mask_close_np1 = sitk.GetArrayFromImage(mask_close1)
        mask_close_rep1 = np.tile(mask_close_np1[ref1,:,:,:], [result_shape1[0],1,1,1])   
        
        # create dataframe 
        # col 1: Specific Ventilation
        ## gaussian filtering, taking lung border into account
        density1 = fl.uniform_filter(mask_close_rep1, size = (0,5,5,5))
        result_gauss1 = fl.gaussian_filter(result_np1 * mask_close_np1, (0,2,2,2), truncate = 1) # gaussian kernel width = 3 
        result_gauss_dens1 = result_gauss1 / (density1 + np.finfo(float).eps)
        ## calculate specific ventilation
        result_rep1 = np.tile(result_gauss_dens1[ref,:,:,:], [result_shape1[0],1,1,1])
        SV1 = (result_rep1 - result_gauss_dens1) / (result_gauss_dens1 + np.finfo(float).eps)
        ## eliminate extreme outliers
        SV1[(SV1<-2) | (SV1>2)] = np.NaN
        SV_np_col1 = np.reshape(SV1, (result_shape1[0] * result_shape1[1] * result_shape1[2] * result_shape1[3]))
        df1 = pd.DataFrame(data=SV_np_col1, columns=["SV"])
        # col 2: Mask
        mask_re1 = np.reshape(mask_close_rep1, (result_shape1[0] * result_shape1[1] * result_shape1[2] * result_shape1[3]))
        df1["mask"] = (mask_re1 != 0)
        # col 3: phase
        tmp1 = np.linspace(1,result_shape1[0],result_shape1[0], dtype = np.uint8)
        phase1 = np.tile(tmp1[:,np.newaxis,np.newaxis,np.newaxis], (1,result_shape1[1],result_shape1[2],result_shape1[3]))
        phase_re1 = np.reshape(phase1, (result_shape1[0] * result_shape1[1] * result_shape1[2] * result_shape1[3]))
        df1['phase'] = phase_re1
        # col 4: scan 1 or 2
        df1['scan'] = np.ones(((result_shape1[0] * result_shape1[1] * result_shape1[2] * result_shape1[3]),1), dtype = np.uint8)
        # col 5: volunteer
        df1['volunteer'] = (ind1 + 1) * np.ones(((result_shape1[0] * result_shape1[1] * result_shape1[2] * result_shape1[3]),1), dtype = np.uint8)
    
        # append dataframe
        df_all1 = df_all1.append(df1, ignore_index=True)
    
    # select based on mask
    df_new1 = df_all1[df_all1["mask"]]
     
    # Step 3: combine data frames
    
    # calculate mean regional ventilation(jacobian-1) of different phase and volunteer
    df_group = df_new.groupby(['volunteer','phase']).mean()
    df_group.reset_index(inplace=True)
    
    df_group1 = df_new1.groupby(['volunteer','phase']).mean()
    df_group1.reset_index(inplace=True)
    
    # calculate difference and average of mean regional ventilation
    df_group['diff'] = df_group['SV'] - df_group1['SV']
    df_group['avg'] = (df_group['SV'] + df_group1['SV']) / 2
    
    # mean and std of difference
    std = 1.96 * np.std(df_group["diff"])
    mean = np.mean(df_group["diff"])
    
    # Bland-Altman plot
    sns.set_context("talk", font_scale = 0.85)
    sns.set_style("ticks")
    sns_plot = sns.relplot(data = df_group, x = "avg", y = "diff", hue = "volunteer", style = "phase", palette = "muted", aspect = 1.2)
    sns_plot.set(xlabel='Average of Total Ventilation of Two Scans ', ylabel='Difference of Total Ventilation between Two Scans', xlim=(-0.1,0.2), ylim = (-0.125, 0.125))
    xlim = [-0.1, 0.2]
    # mean and 1.96 std lines
    plt.plot(xlim, [mean+std, mean+std],'k--', xlim, [mean-std, mean-std],'k--')
    plt.plot(xlim, [mean, mean])
    # add text to the lines
    plt.text(0.2, mean, 'Mean \n %.2e' % mean, fontsize=14, horizontalalignment='right', verticalalignment='center', multialignment='right')
    plt.text(0.2, mean+std, '+1.96 SD \n %.2e' % (mean+std), fontsize=14, horizontalalignment='right', verticalalignment='center', multialignment='right')
    plt.text(0.2, mean-std, '-1.96 SD \n %.2e' % (mean-std), fontsize=14, horizontalalignment='right', verticalalignment='center', multialignment='right')
    # save figure
    sns_plot.savefig(output_dir + 'SV_BlandAltman_'+reg[:-1]+'.png', bbox_inches='tight', dpi = 300)
    
    
    # linear regression plot
    df_group['SV_scan2'] = df_group1['SV']
    # calculate spearman r, slope
    slope, intercept, r, p, se = stats.linregress(df_group['SV'], df_group1['SV'])
    
    # linear regression line plot
    sns.set_context("talk", font_scale = 0.85)
    g = sns.relplot(data = df_group, x = "SV", y = "SV_scan2", hue = "volunteer", style = "phase", palette = "muted", aspect = 1.2)
    sns.regplot(x=df_group['SV'], y=df_group1['SV'], scatter = False)
    g.set(xlim=(-0.05, 0.2), ylim=(-0.05, 0.2),xlabel = 'Total Ventilation of Scan 1', ylabel = 'Total Ventilation of Scan 2')
    g.fig.text(0.7,0.85,'R$^2$={0:.2f}, p={1:.2e} \n TV2 = {2:.2f} + {3:.2f}TV1'.format(r,p,intercept,slope), horizontalalignment='right')
    plt.plot([-0.05, 0.2], [-0.05, 0.2], 'k--')
    g.savefig(output_dir + 'SV_LinearRegression_'+reg[:-1]+'.png', bbox_inches='tight', dpi = 300)

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
    
    bland_altman_jacobian(output_dir_list1, mask_dir_list1, output_dir_list2, mask_dir_list2, reg_list[0])
    bland_altman_jacobian(output_dir_list1, mask_dir_list1, output_dir_list2, mask_dir_list2, reg_list[2])
    bland_altman_sv(output_dir_list1, mask_dir_list1, output_dir_list2, mask_dir_list2, reg_list[0])
    bland_altman_sv(output_dir_list1, mask_dir_list1, output_dir_list2, mask_dir_list2, reg_list[2])