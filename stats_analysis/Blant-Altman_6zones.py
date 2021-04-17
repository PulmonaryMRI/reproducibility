#!/usr/bin/env python
# coding: utf-8

import seaborn as sns
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

def bland_altman_jacobian_6zone(dir_list, mask_list, dir_list1, mask_list1, reg):
    """
    Cross subject Bland-Altman plot of jacobian determinant of each zone

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
    df_all = pd.DataFrame(columns = ['Jac','mask','phase','scan','volunteer','zone'])
    
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
        # col 6: grid mask
        # grid mask
        gridmask = grid_mask(mask_path_close)
        zone_list = np.array(['Background','Lower Left','Lower Right','Middle Left','Middle Right','Upper Left','Upper Right'])    
        df['zone'] = zone_list[np.reshape(np.tile(np.uint8(gridmask[np.newaxis]), [Jac_shape[0],1,1,1]), (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))]
 
        # append dataframe
        df_all = df_all.append(df, ignore_index=True)
    
    # select based on mask
    df_new = df_all[df_all["mask"]]
    df_new = df_new[df_new["zone"]!='Background']
    # Step 2: import data from all scan 2's

    # initialize data frame
    df_all1 = pd.DataFrame(columns = ['Jac','mask','phase','scan','volunteer','zone'])
    
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
        # col 6: grid mask
        gridmask1 = grid_mask(mask_path_close1)
        df1['zone'] = zone_list[np.reshape(np.tile(np.uint8(gridmask1[np.newaxis]), [Jac_shape1[0],1,1,1]), (Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]))]
 
        # append dataframe
        df_all1 = df_all1.append(df1, ignore_index=True)
    
    # select based on mask
    df_new1 = df_all1[df_all1["mask"]]
    df_new1 = df_new1[df_new1["zone"]!='Background']
    
    # Step 3: combine data frames
    
    # calculate mean regional ventilation(jacobian-1) of different phase and volunteer
    df_group = df_new.groupby(['volunteer','phase','zone']).mean()
    df_group.reset_index(inplace=True)
    
    df_group1 = df_new1.groupby(['volunteer','phase','zone']).mean()
    df_group1.reset_index(inplace=True)
    
    # calculate difference and average of mean regional ventilation
    df_group['diff'] = df_group['Jac'] - df_group1['Jac']
    df_group['avg'] = (df_group['Jac'] + df_group1['Jac']) / 2
    
    # save total Bland-Altman
    sns.set_context("talk", font_scale = 1.2)
    sns.set_style("ticks")
    g = sns.relplot(data = df_group, x = "avg", y = "diff", hue = "volunteer", style = "phase", col = "zone", col_wrap = 2, palette = "muted", aspect = 1.2)
    g.map_dataframe(draw_mean_std, xlim = [-0.1,0.3])    
    g.set(xlabel='Average of Total Ventilation \n of Two Scans ', ylabel='Difference of Total Ventilation \n between Two Scans', xlim=(-0.1,0.3), ylim = (-0.125, 0.125))
    g.savefig(output_dir + 'BlandAltman_6zone_all.png', bbox_inches='tight', dpi=300)
    
    # save individual Bland-Altman
    for ind in range(len(zone_list)-1):    
        # mean and std of difference
        std = 1.96 * np.std(df_group[df_group["zone"]==zone_list[ind+1]]["diff"])
        mean = np.mean(df_group[df_group["zone"]==zone_list[ind+1]]["diff"])
        # Bland-Altman plot
        sns.set_context("talk", font_scale = 1.2)
        sns.set_style("ticks")
        sns_plot = sns.relplot(data = df_group[df_group["zone"]==zone_list[ind+1]], x = "avg", y = "diff", hue = "volunteer", style = "phase", palette = "muted", aspect = 1.2)
        sns_plot.set(xlabel='Average of Total Ventilation \n of Two Scans ', ylabel='Difference of Total Ventilation \n between Two Scans', xlim=(-0.1,0.3), ylim = (-0.125, 0.125))
        xlim = [-0.1, 0.3]
        # mean and 1.96 std lines
        plt.plot(xlim, [mean+std, mean+std],'k--', xlim, [mean-std, mean-std],'k--')
        plt.plot(xlim, [mean, mean])
        # add text to the lines
        plt.text(0.3, mean, 'Mean \n %.2e' % mean, fontsize=20, horizontalalignment='right', verticalalignment='center', multialignment='right')
        plt.text(0.3, mean+std, '+1.96 SD \n %.2e' % (mean+std), fontsize=20, horizontalalignment='right', verticalalignment='center', multialignment='right')
        plt.text(0.3, mean-std, '-1.96 SD \n %.2e' % (mean-std), fontsize=20, horizontalalignment='right', verticalalignment='center', multialignment='right')
        # save figure
        sns_plot.savefig(output_dir + 'JD_BlandAltman_6zone_'+str(ind)+'.png', bbox_inches='tight', dpi = 300)

def draw_mean_std(data, xlim = [0,1], **kws):
    std = 1.96 * data.std()['y']
    mean = data.mean()['y']
    ax = plt.gca()
    ax.plot(xlim, [mean+std, mean+std],'k--', xlim, [mean-std, mean-std],'k--')
    ax.plot(xlim, [mean, mean])
    # add text to the lines
    ax.text(0.3, mean, 'Mean \n %.2e' % mean, fontsize=20, horizontalalignment='right', verticalalignment='center', multialignment='right')
    ax.text(0.3, mean+std, '+1.96 SD \n %.2e' % (mean+std), fontsize=20, horizontalalignment='right', verticalalignment='center', multialignment='right')
    ax.text(0.3, mean-std, '-1.96 SD \n %.2e' % (mean-std), fontsize=20, horizontalalignment='right', verticalalignment='center', multialignment='right')

def split_violin_jacobian_6zone(output_dir, mask_path_close, output_dir1, mask_path_close1):
    # load Jacobian determinant
    JacDet = sitk.ReadImage(output_dir + "JacDet_4d.nii")
    JacDet_np = sitk.GetArrayFromImage(JacDet)
    Jac_shape = np.shape(JacDet_np)
    
    # load tight mask
    ref = 0
    mask_close = sitk.ReadImage(mask_path_close)
    mask_close_np = sitk.GetArrayFromImage(mask_close)
    mask_close_rep = np.tile(mask_close_np[ref,:,:,:], [Jac_shape[0],1,1,1])
    
    # identify 6 zones
    gridmask = grid_mask(mask_path_close)   
    zone_list = np.array(['Background','Lower Left','Lower Right','Middle Left','Middle Right','Upper Left','Upper Right'])
    
    # create dataframe 
    # col 1: Jacobian determiant
    JacDet_np_col = np.reshape(JacDet_np - 1, (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
    df = pd.DataFrame(data=JacDet_np_col, columns=["Jac"])
    # col 2: Mask
    mask_re = np.reshape(mask_close_rep, (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
    df["mask"] = (mask_re != 0)
    # col 3: phase
    tmp = np.linspace(1,12,12, dtype = np.uint8)
    phase = np.tile(tmp[:,np.newaxis,np.newaxis,np.newaxis], (1,Jac_shape[1],Jac_shape[2],Jac_shape[3]))
    phase_re = np.reshape(phase, (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
    df['phase'] = phase_re
    # col 4: scan 1 or 2
    df['scan'] = np.ones(((Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]),1), dtype = np.uint8)
    # col 5: grid
    df['zone'] = zone_list[np.reshape(np.tile(np.uint8(gridmask[np.newaxis]), [Jac_shape[0],1,1,1]), (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))]
    
    # select based on mask
    df_new = df[df["mask"]]
    df_grid = df_new[df_new["zone"]!='Background']
    
    # group dataframe
    df_group = df_grid.groupby(['zone','phase']).mean()
    df_group.reset_index(inplace=True)
    df_group
      
    # g = sns.lineplot(data=df_grid, x="phase", y="Jac", hue="zone")
       
    # load Jacobian determinant 2
    JacDet1 = sitk.ReadImage(output_dir1 + "JacDet_4d.nii")
    JacDet_np1 = sitk.GetArrayFromImage(JacDet1)
    Jac_shape1 = np.shape(JacDet_np1)
    # load tight mask
    ref = 0
    mask_close1 = sitk.ReadImage(mask_path_close1)
    mask_close_np1 = sitk.GetArrayFromImage(mask_close1)
    mask_close_rep1 = np.tile(mask_close_np1[ref,:,:,:], [np.shape(mask_close_np1)[0],1,1,1])
    
    # grid mask 1
    gridmask1 = grid_mask(mask_path_close1)
 
    # create dataframe 2
    # col 1: Jacobian determiant
    JacDet_np_col1 = np.reshape(JacDet_np1-1, (Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]))
    df1 = pd.DataFrame(data=JacDet_np_col1, columns=["Jac"])
    # col 2: Mask
    mask_re1 = np.reshape(mask_close_rep1, (Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]))
    df1["mask"] = (mask_re1 != 0)
    # col 3: phase
    tmp1 = np.linspace(1,12,12, dtype = np.uint8)
    phase1 = np.tile(tmp1[:,np.newaxis,np.newaxis,np.newaxis], (1,Jac_shape1[1],Jac_shape1[2],Jac_shape1[3]))
    np.shape(phase)
    phase_re1 = np.reshape(phase1, (Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]))
    df1['phase'] = phase_re1
    # col 4: scan 1 or 2 
    df1['scan'] = 2 * np.ones(((Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]),1), dtype = np.uint8)
    # col 5: grid
    df1['zone'] = zone_list[np.reshape(np.tile(np.uint8(gridmask1[np.newaxis]), [Jac_shape1[0],1,1,1]), (Jac_shape1[0] * Jac_shape1[1] * Jac_shape1[2] * Jac_shape1[3]))]
    # select based on mask
    df_new1 = df1[df1["mask"]]
    df_grid1 = df_new1[df_new1["zone"]!='Background']

    
    # combine 2 data frame and sanity check
    df_concat = pd.concat([df_grid, df_grid1], ignore_index=True, sort=False)
    return df_concat
    
    # # group data frame 2
    # df_group1 = df_grid1.groupby(['zone','phase']).mean()
    # df_group1.reset_index(inplace=True)
    
    # # compute diff and avg    
    # df_group['diff'] = df_group['Jac'] - df_group1['Jac']
    # df_group['avg'] = (df_group['Jac'] + df_group1['Jac']) / 2
    
    # # compute mean and standard deviation
    # std = 1.96 * np.std(df_group["diff"])
    # mean = np.mean(df_group["diff"])
    
    # # Bland-Altman plot of each subject
    # sns.set_context("talk", font_scale = 0.85)
    # sns.set_style("ticks")
    # sns_plot = sns.relplot(data = df_group, x = "avg", y = "diff", style = "phase", hue = "zone", palette = "muted", aspect = 1.2)
    # sns_plot.set(xlabel='Average of Total Ventilation of Two Scans ', ylabel='Difference of Total Ventilation between Two Scans', xlim=(-0.1,0.3), ylim = (-0.125, 0.125))
    # xlim = [-0.1, 0.3]
    # # mean and 1.96 std lines
    # plt.plot(xlim, [mean+std, mean+std],'k--', xlim, [mean-std, mean-std],'k--')
    # plt.plot(xlim, [mean, mean])
    # # add text to the lines
    # plt.text(0.3, mean, 'Mean \n %.2e' % mean, fontsize=14, horizontalalignment='right', verticalalignment='center', multialignment='right')
    # plt.text(0.3, mean+std, '+1.96 SD \n %.2e' % (mean+std), fontsize=14, horizontalalignment='right', verticalalignment='center', multialignment='right')
    # plt.text(0.3, mean-std, '-1.96 SD \n %.2e' % (mean-std), fontsize=14, horizontalalignment='right', verticalalignment='center', multialignment='right')
    # # save figure
    # sns_plot.savefig(output_dir + 'JD_BlandAltman_6zone_vol.png', bbox_inches='tight', dpi = 300)
    
    # split violin plot of each zone
    # for ind in range(1,7):
    #     plt.figure()
    #     sns.set_context("talk")
    #     sns_plot = sns.violinplot(data=df_concat[df_concat['zone']==zone_list[ind]], y="Jac", x="phase", hue="scan",
    #                    split=True, inner="quart", linewidth=1.5, bw = .2)
    #     sns_plot.set_ylim(-0.2, 0.6)
    #     sns_plot.set(ylabel='Regional Ventilation (Jac - 1)')
    sns.set_context("talk")
    g = sns.catplot(data=df_concat, x="phase", y="Jac", kind = "violin", col="zone", margin_titles=True, hue="scan", 
                    split=True, inner="quart", linewidth=1.5, bw = .2, ylim=(-0.2, 0.6), col_order=['Lower Left','Lower Right','Middle Left','Middle Right','Upper Left','Upper Right'])
    g.savefig(output_dir + 'JD_split_violin_6zone.png', bbox_inches='tight')

def grid_mask(mask_path_close):
    # load mask
    ref = 0
    mask_close = sitk.ReadImage(mask_path_close)
    mask_close_np = sitk.GetArrayFromImage(mask_close)
    mask_shape = np.shape(mask_close_np)          
    
    # identify extremes
    X,Y,Z = np.meshgrid(range(mask_shape[1]), range(mask_shape[2]), range(mask_shape[3]), indexing = 'ij')
    xmax = np.max(X[mask_close_np[ref,:,:,:]>0])
    xmin = np.min(X[mask_close_np[ref,:,:,:]>0])
    ymax = np.max(Y[mask_close_np[ref,:,:,:]>0])
    ymin = np.min(Y[mask_close_np[ref,:,:,:]>0])
    zmax = np.max(Z[mask_close_np[ref,:,:,:]>0])
    zmin = np.min(Z[mask_close_np[ref,:,:,:]>0])
    print(xmax,xmin,ymax,ymin,zmax,zmin)    
    
    # number of grids and grid end points
    xn,yn,zn = 3,1,2; # for 6-zone 3,1,2
    xgrid = np.uint8(np.linspace(xmin, xmax, xn+1))
    ygrid = np.uint8(np.linspace(ymin, ymax, yn+1))
    zgrid = np.uint8(np.linspace(zmin, zmax, zn+1))
    print(xgrid,ygrid,zgrid)
    
    # gridmask
    gridmask = np.zeros(mask_shape[1:])
    ind = 1
    for i in range(xn):
        for j in range(yn):
            for k in range(zn):
                gridmask[xgrid[i]:xgrid[i+1], ygrid[j]:ygrid[j+1], zgrid[k]:zgrid[k+1]] = ind
                ind += 1
                
    return gridmask

def jacobian_coefficient_of_variation_6zone(output_dir, mask_dir, output_dir1):
    print(output_dir)
    
    # load jacobian determinant of scan 1
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
    
    print(output_dir1)
    JacDet1 = sitk.ReadImage(output_dir1 + "JacDet_4d_tx.nii")
    JacDet_np1 = sitk.GetArrayFromImage(JacDet1)
    
    # col 4: Jacobian determiant 2 tx
    JacDet_np_col1 = np.reshape(np.log(JacDet_np1), (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))
    df["Jac2_lg"] = JacDet_np_col1

    # identify 6 zones
    gridmask = grid_mask(mask_path_close)   
    zone_list = np.array(['Background','Lower Left','Lower Right','Middle Left','Middle Right','Upper Left','Upper Right'])
    # col 5: grid
    df['zone'] = zone_list[np.reshape(np.tile(np.uint8(gridmask[np.newaxis]), [Jac_shape[0],1,1,1]), (Jac_shape[0] * Jac_shape[1] * Jac_shape[2] * Jac_shape[3]))]
    

    # select based on mask
    df_new = df[df["mask"]]
    df_new = df_new[df_new["zone"]!='Background']
    
    # within subject CV, log method
    df_new['diff'] =  (df_new['Jac1_lg'] - df_new['Jac2_lg']) ** 2 / 2
    df_group = df_new.groupby(['phase','zone']).mean()
    df_group.reset_index(inplace=True)
    df_group['CV'] = np.exp(np.sqrt(df_group['diff'])) - 1
    
    return df_group

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
    
    reg1 = '3DnT_BSpline/'
    
    df_all_jd = pd.DataFrame()
    
    for ind in range(len(output_dir_list1)):        
        output_dir1 = output_dir_list1[ind] + reg1
        mask_path_close1 = mask_dir_list1[ind] + "lung_mask_close.nii"
        output_dir2 = output_dir_list2[ind] + reg1
        mask_path_close2 = mask_dir_list2[ind] + "lung_mask_close.nii"
        
        df_tmp = split_violin_jacobian_6zone(output_dir1, mask_path_close1, output_dir2, mask_path_close2)
        df_tmp['volunteer'] = (ind+1)
        df_tmp['reg'] = 'JD_3DnT'  
        df_all_jd = df_all_jd.append(df_tmp, ignore_index=True)

    
    
    # violin plots
    plt.close('all')
    sns.set_context("talk", font_scale=1.75, rc={"lines.linewidth": 2.5})
    g1 = sns.catplot(data=df_all_jd, x="phase", y="Jac", kind = "violin", col = "zone", row="volunteer", margin_titles=True, hue="scan", split=True, inner="quart", linewidth=1.5, bw = .2, col_order = ['Lower Left','Lower Right','Middle Left','Middle Right','Upper Left','Upper Right'], palette='deep', aspect=1.2)
    g1.map_dataframe(sns.pointplot, x="phase", y="Jac",ci=None, hue="scan", dodge=0.5, estimator=np.median, palette = "tab10",markers =['.','.'],scale = 0.8)
    g1.set(ylim=(-0.5,0.8), xlabel = 'phase', ylabel = "Regional Ventilation")
    g1.savefig(output_dir1 + 'split_violin_6zone_all.png', bbox_inches='tight', dpi=300)
    
    
    # # bland_altman
    # bland_altman_jacobian_6zone(output_dir_list1, mask_dir_list1, output_dir_list2, mask_dir_list2, reg1)
    
    # coefficient of variation 
    # df_all = pd.DataFrame()
    # for ind in range(len(output_dir_list1)):
    #     output_dir1 = output_dir_list1[ind] + reg1
    #     output_dir2 = output_dir_list2[ind] + reg1
    #     mask_dir = mask_dir_list1[ind]
        
    #     df_tmp = jacobian_coefficient_of_variation_6zone(output_dir1, mask_dir, output_dir2)
    #     df_tmp['vol'] = (ind + 1) * np.ones(len(df_tmp['CV']))
    #     # append dataframe
    #     df_all = df_all.append(df_tmp, ignore_index=True)
        
    # sns.set_context("talk", font_scale = 1.2)
    # g = sns.catplot(data=df_all, x="phase", y="CV", hue="zone", kind = "box", palette = 'colorblind', legend_out = True, height = 8, aspect = 1.2, row_order = ['Lower Left','Lower Right','Middle Left','Middle Right','Upper Left','Upper Right'])
    # plt.ylabel("coefficient of variation")
    # plt.xlabel("respiratory phase")
    # g.savefig(output_dir_list1[-1] + reg1 + 'coefficient_of_variation_6zone.png', bbox_inches='tight', dpi = 300)