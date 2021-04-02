#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:14:07 2021

@author: ftan1

elastix registration related functions
"""

import SimpleITK as sitk
import numpy as np
import sigpy.plot as pl
import matplotlib.pyplot as plt
import os, glob
import time
import sys
import scipy.ndimage.filters as fl

def elastix_reg(image_path, mask_path_dilate, output_dir):
    """
    Conduct 4D cyclic Elastix registration, register to a intermediate brathing position
    Command line equivalent:
        elastix -f image_path -m mask_path_dilate -p parameter_file -out output_dir
    
    Parameters
    ----------
    image_path : string 
        phase-resolved image (.nii format) path.
    mask_path_dilate : string
        path to the dilated mask calculated by the antspynet segmentation.
    output_dir : string
        output directory for transform parameter files.

    Returns
    -------
    none.

    """
    # mark the starting time
    start_time = time.time()
    
    # load image
    image = sitk.ReadImage(image_path)
    npImage = sitk.GetArrayFromImage(image)
    #pl.ImagePlot(npImage, x = -1, y = -3)
    
    # load mask (4D)    
    mask = sitk.ReadImage(mask_path_dilate)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.SetSpacing(image.GetSpacing())
    
    # load parameter file
    para_fwd_path = 'par000.forward.txt'
    para_fwd = sitk.ReadParameterFile(para_fwd_path)
    para_inv_path = 'par001.inverse.txt'
    para_inv = sitk.ReadParameterFile(para_inv_path)
    
    # set filter
    elastixImageFilter = sitk.ElastixImageFilter()
    
    # set fixed and moving image, in 4d registration, fixed image is a dummy variable
    elastixImageFilter.SetFixedImage(image)
    elastixImageFilter.SetMovingImage(image)
    
    # set mask
    elastixImageFilter.SetFixedMask(mask)
    elastixImageFilter.SetMovingMask(mask)
    
    # set parameter files 
    elastixImageFilter.SetParameterMap(para_fwd)
    elastixImageFilter.AddParameterMap(para_inv)
    
    # set output dir
    elastixImageFilter.SetOutputDirectory(output_dir)
    
    # run
    elastixImageFilter.Execute()
    
    # save result image
    resultImage = elastixImageFilter.GetResultImage()
    sitk.WriteImage(resultImage, output_dir + "result0.nii")
    
    # display result image
    resultImage_np = sitk.GetArrayFromImage(resultImage)
    #pl.ImagePlot(resultImage_np, x = -1, y = -3)
    
    # remove unnecessary files
    for file in glob.glob(output_dir + 'IterationInfo*'):
        os.remove(file)
    
    # caclulate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('elastix_reg done. elapsed time:', elapsed_time, 's')


def combine_par(output_dir):
    """
    Combine parameters created by elastix_reg(). adapted from combine.py

    Parameters
    ----------
    output_dir : string
        input and output directory for transform parameter files, same as in elastix_reg().

    Returns
    -------
    none.

    """ 
    #start time
    start_time = time.time()
    
    # set input/output file paths
    infile0 = output_dir + 'TransformParameters.0.txt'
    infile1 = output_dir + 'TransformParameters.1.txt'
    outfile0 = output_dir +'TransformParameters.fwd.txt'
    outfile1 = output_dir +'TransformParameters.inv.txt'
    
    # define reference frame for registration
    ref = 0
    spacing = 1
    
    # Open parameter file 0 and search for GridSpacing and GridOrigin line
    text_filein0 = open( infile0, "r" )
    for line in text_filein0:
      if line.find( "(GridOrigin " ) == 0:
        origin_str = line
      elif line.find( "(GridSpacing " ) == 0:
        spacing_str = line
    text_filein0.close()
    
    # Extract time point origin from line
    origin_split = origin_str.strip().split(' ')
    origin_split = origin_split[ len( origin_split ) - 1 ].split(')')
    old_origin = float( origin_split[ 0 ] )
    
    # Extract time point spacing from line
    spacing_split = spacing_str.strip().split(' ')
    spacing_split = spacing_split[ len( spacing_split ) - 1 ].split(')')
    old_spacing = float( spacing_split[ 0 ] )
    
    
    print("Original grid origin in time dimension: " + str( old_origin ))
    print("Original grid spacing in time dimension: " + str( old_spacing ))
    print("")
    
    # Determine new grid origin
    new_origin = ref - ( spacing / old_spacing ) * ( ref - old_origin )
    print( "New grid origin in time dimension: " + str( new_origin ))
    
    # Recompose origin and spacing lines
    new_origin_string = origin_str.strip().split(' ')
    new_origin_string.pop()
    new_origin_string = " ".join( new_origin_string ) + " " + str( new_origin ) + ")\n"
    new_spacing_string = spacing_str.strip().split(' ')
    new_spacing_string.pop()
    new_spacing_string = " ".join( new_spacing_string ) + " " + str( spacing ) + ")\n"
    
    # Reopen text file, replace origin and spacing and write to output file 0
    text_filein0 = open( infile0, "r" )
    text_fileout0 = open( outfile0, "w" )
    for line in text_filein0:
      if line.find( "(GridOrigin " ) == 0:
        # Write new origin line
        text_fileout0.write( new_origin_string )
      elif line.find( "(GridSpacing " ) == 0:
        # Write new spacing line
        text_fileout0.write( new_spacing_string )
      elif line.find( "(InitialTransformParametersFileName " ) == 0:
        # Remove initial transform
        text_fileout0.write( "(InitialTransformParametersFileName \"NoInitialTransform\")\n" )
      else:
        # Write line read from input file (no change)
        text_fileout0.write( line )
    text_filein0.close()
    text_fileout0.close()
    
    # Open parameter file 1 and search for GridSize
    text_filein1 = open( infile1, "r" )
    for line in text_filein1:
      if line.find("(GridSize") == 0:
        grid_str = line
        grid_split = grid_str.strip().split(' ')
        grid_split[-1] = grid_split[-1].replace(')','')
        grid_split = grid_split[1:]
        grid_float = [float(s) for s in grid_split]
        grid_all = int(grid_float[0] * grid_float[1] * grid_float[2] * grid_float[3])
        num_phase = int(grid_float[3])
    text_filein1.close()
    
    # Replace initial transform parameter filename
    text_filein1 = open( infile1, "r" )
    text_fileout1 = open( outfile1, "w" )
    for line in text_filein1:
      if line.find( "(InitialTransformParametersFileName " ) == 0:
        # Set initial transform filename
        text_fileout1.write( "(InitialTransformParametersFileName \"" + outfile0  + "\")\n" )
      elif line.find("(TransformParameters ") == 0:
        # extract b-spline parameters, arrangment : x (Px*Py*Pz), y(Px*Py*Pz), z(Px*Py*Pz), t(Px*Py*Pz)
        transPar_str = line
        transPar_split = transPar_str.strip().split(' ')
        transPar_split[-1] = transPar_split[-1].replace(')','')
        transPar_split = transPar_split[1:]
        num_grid3d = int(grid_all / num_phase)   
        str_seg = transPar_split[(ref*num_grid3d):((ref+1)*num_grid3d)] * num_phase + transPar_split[(grid_all+(ref*num_grid3d)): (grid_all + (ref+1)*num_grid3d)] * num_phase + transPar_split[(grid_all*2+(ref*num_grid3d)): (grid_all*2 + (ref+1)*num_grid3d)] * num_phase + transPar_split[(grid_all*3+(ref*num_grid3d)): (grid_all*3 + (ref+1)*num_grid3d)] * num_phase
        #str_seg = ""
        #str_seg = [str_seg + transPar_split[((ref*num_grid3d)+grid*i):((ref+1)*num_grid3d+grid*i)] * num_phase for i in range(4)]
        str_joined = ' '.join(str_seg)
        text_fileout1.write("(TransformParameters " + str_joined + ")\n")
      else:
        # Write line read from input file (no change)
        text_fileout1.write( line )
    text_filein1.close()
    text_fileout1.close()
    
    # caclulate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('combine_par done. elapsed time:', elapsed_time, 's')

def transformix_reg(output_dir, image_path):

    """
    Compute registered image and jacobian determinant w.r.t reference frame by Transformix
    Command line equivalent:
        transformix -tp TransformParameters.inv.txt -in image_path -out output_dir -jac all -def all

    Parameters
    ----------
    output_dir : string
        output directory for registered image and jacobian determinant.
    image_path : string
        phase-resolved image (.nii format) path.

    Returns
    -------
    none.

    """
    # start time
    start_time = time.time()
    
    # read parameter files
    trans_dir_inv = output_dir +'TransformParameters.inv.txt'
    trans_inv = sitk.ReadParameterFile(trans_dir_inv)
    
    trans_dir_fwd = output_dir +'TransformParameters.fwd.txt'
    trans_fwd = sitk.ReadParameterFile(trans_dir_fwd)

    # load image
    image = sitk.ReadImage(image_path)

    # initialize transformix
    transformixImageFilter = sitk.TransformixImageFilter()
    
    # set moving image
    transformixImageFilter.SetMovingImage(image)
    
    # set parameter
    transformixImageFilter.SetTransformParameterMap(trans_fwd)
    transformixImageFilter.AddTransformParameterMap(trans_inv)
    
    # set options, compute deformation field and Jac det
    transformixImageFilter.ComputeDeformationFieldOn()
    transformixImageFilter.ComputeDeterminantOfSpatialJacobianOn()
    
    # set output dir
    transformixImageFilter.SetOutputDirectory(output_dir)

    # run
    transformixImageFilter.Execute()

    # get results, jacobian det doesn't have a function, deformation field cannot be accessed
    tranResultImage = transformixImageFilter.GetResultImage()
    #JacDet = sitk.ReadImage(output_dir + "spatialJacobian.nii")
    
    # save results, rename spatialJacobian for consistancy
    sitk.WriteImage(tranResultImage, output_dir + "result_4d.nii")
    os.rename(output_dir + "spatialJacobian.nii", output_dir + "JacDet_4d.nii")

    # display registered image
    tranResultImage_np = sitk.GetArrayFromImage(tranResultImage)
    #pl.ImagePlot(tranResultImage_np, x = -1, y = -3)
    
    # remove unnecessary files
    for file in glob.glob(output_dir + 'TransformParameters*'):
        os.remove(file)
    
    # caclulate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('transformix_reg done. elapsed time:', elapsed_time, 's')

def display_jacobian_4d(output_dir, mask_path_close):

    # Display Jacobian determiant with Sigpy
    JacDet = sitk.ReadImage(output_dir + "JacDet_4d.nii")
    JacDet_np = sitk.GetArrayFromImage(JacDet)
    JacDet_np = (JacDet_np - 1) # regional ventilation, volume change in ml / ml
    
    # plot grayscale image
    tranResultImage = sitk.ReadImage(output_dir + "result_4d.nii")
    tranResultImage_np = sitk.GetArrayFromImage(tranResultImage)
    
    # load tight mask
    ref = 0
    mask_close = sitk.ReadImage(mask_path_close)
    mask_close_np = sitk.GetArrayFromImage(mask_close)
    mask_close_rep = np.tile(mask_close_np[ref,:,:,:], [np.shape(JacDet_np)[0],1,1,1])
    
    # overlay masked jacobian
    JacDet_masked = np.ma.masked_where(mask_close_rep==0, JacDet_np)
    pl.ImagePlot(tranResultImage_np, x = -1, y = -3,  overlay = JacDet_masked, alpha=0.5)


def display_jacobian_2d(output_dir, mask_path_close):
    # load jacobian determinant
    JacDet = sitk.ReadImage(output_dir + "JacDet_4d.nii")
    JacDet_np = sitk.GetArrayFromImage(JacDet)
    JacDet_np = (JacDet_np-1) # regional ventilation, volume change in ml / ml
    
    # load registered image
    tranResultImage = sitk.ReadImage(output_dir + "result_4d.nii")
    tranResultImage_np = sitk.GetArrayFromImage(tranResultImage)
    ref = 0
    # display overlaid Jacobian
    for phase in range(np.shape(JacDet_np)[0]):
        fig = plt.figure()
        slc = 64
        # load tight mask
        mask_close = sitk.ReadImage(mask_path_close)
        mask_close_np = sitk.GetArrayFromImage(mask_close)
        mask_close_rep = np.tile(mask_close_np[ref,:,:,:], [np.shape(JacDet_np)[0],1,1,1])
    
        # plot grayscale image
        fig = plt.imshow(tranResultImage_np[phase,:,slc,:], cmap = 'gray')
        fig.set_clim(np.amin(tranResultImage_np[phase,:,slc,:]), 0.6 * np.amax(tranResultImage_np[phase,:,slc,:]))
        
        # overlay masked jacobian
        JacDet_masked = np.ma.masked_where(mask_close_rep==0, JacDet_np)
        fig = plt.imshow(JacDet_masked[phase,:,slc,:], cmap = 'viridis', alpha = 0.8, vmin = -0, vmax = 0.5)
        plt.axis('off')
        plt.colorbar(fig, extend = 'both', label='Regional Ventilation [ml/ml]', shrink = 0.8)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.show()
        
        plt.savefig(output_dir + 'JD_3DnT_p' + str(phase) + '.png', transparent=True)

def display_sv_4d(output_dir, mask_path_close):
    # Display Specific Ventilation with Sigpy
    
    # lgrayscaleoad  image
    tranResultImage = sitk.ReadImage(output_dir + "result_4d.nii")
    tranResultImage_np = sitk.GetArrayFromImage(tranResultImage)
    
    # calculate specific ventilation 
    ref = 0
    tranResultImage_rep = np.tile(tranResultImage_np[ref,:,:,:], [np.shape(tranResultImage_np)[0],1,1,1])
    SV = (tranResultImage_rep - tranResultImage_np) / (tranResultImage_np + np.finfo(float).eps)
    
    # load tight mask
    mask_close = sitk.ReadImage(mask_path_close)
    mask_close_np = sitk.GetArrayFromImage(mask_close)
    mask_close_rep = np.tile(mask_close_np[ref,:,:,:], [np.shape(mask_close_np)[0],1,1,1])
    
    # overlay masked jacobian
    SV_masked = np.ma.masked_where((mask_close_rep==0)|(SV<-2)|(SV>2), SV)
    pl.ImagePlot(tranResultImage_np, x = -1, y = -3,  overlay = SV_masked,alpha=0.5)
    
def display_sv_2d(output_dir, mask_path_close):
    # load registered image
    tranResultImage = sitk.ReadImage(output_dir + "result_4d.nii")
    tranResultImage_np = sitk.GetArrayFromImage(tranResultImage)
    ref = 0
    result_shape = np.shape(tranResultImage_np)
    
    # load tight mask
    mask_close = sitk.ReadImage(mask_path_close)
    mask_close_np = sitk.GetArrayFromImage(mask_close)
    mask_close_rep = np.tile(mask_close_np[ref,:,:,:], [np.shape(tranResultImage_np)[0],1,1,1])
    
    # col 1: Specific Ventilation
    ## gaussian filtering, taking lung border into account
    density = fl.uniform_filter(mask_close_rep, size = (0,5,5,5))
    result_gauss = fl.gaussian_filter(tranResultImage_np * mask_close_rep, (0,2,2,2), truncate = 1) # gaussian kernel width = 3 
    result_gauss_dens = result_gauss / (density + np.finfo(float).eps)
    ## calculate specific ventilation
    result_rep = np.tile(result_gauss_dens[ref,:,:,:], [result_shape[0],1,1,1])
    SV = (result_rep - result_gauss_dens) / (result_gauss_dens + np.finfo(float).eps)
    ## eliminate extreme outliers
    SV[(SV<-2) | (SV>2)] = np.NaN
    
    SV_masked = np.ma.masked_where(mask_close_rep==0, SV)
    # display overlaid Jacobian
    for phase in range(result_shape[0]):
        fig = plt.figure()
        slc = 64
    
        # plot grayscale image
        fig = plt.imshow(tranResultImage_np[phase,:,slc,:], cmap = 'gray')
        fig.set_clim(np.amin(tranResultImage_np[phase,:,slc,:]), 0.6 * np.amax(tranResultImage_np[phase,:,slc,:]))
        
        # overlay masked SV        
        fig = plt.imshow(SV_masked[phase,:,slc,:], cmap = 'viridis', alpha = 0.8, vmin = -0, vmax = 0.5)
        plt.axis('off')
        plt.colorbar(fig, extend = 'both', label='Specific Ventilation [ml/ml]', shrink = 0.8)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.show()
        
        plt.savefig(output_dir + 'SV_3DnT_p' + str(phase) + '.png', transparent=True)
        
if __name__ == '__main__':
    
    """
    Discription
    main script for 4D cyclic elastix/transformix registration

    Example usage in terminal:
    python elastix.py 2020-08-20_vo P56320
    """

    # take input
    date = str(sys.argv[1]) #'2020-08-20_vo'
    pfile = str(sys.argv[2]) #'P56320'
    
    # directories
    output_dir = '/data/larson4/UTE_Lung/' + date + '/reg/' + pfile + '/3DnT_BSpline/'
    image_path = '/data/larson4/UTE_Lung/' + date + '/cfl/' + pfile + '/MRI_Raw_pr_rec_v3.nii'
    mask_path_dilate = '/data/larson4/UTE_Lung/' + date + '/seg/' + pfile + '/lung_mask_dilate.nii'
    mask_path_close = '/data/larson4/UTE_Lung/' + date + '/seg/' + pfile + '/lung_mask_close.nii'
    
    # registration
    elastix_reg(image_path, mask_path_dilate, output_dir)
    combine_par(output_dir)
    transformix_reg(output_dir, image_path)
    display_jacobian_4d(output_dir, mask_path_close)
    display_sv_4d(output_dir, mask_path_close)
    display_sv_2d(output_dir, mask_path_close)
    display_jacobian_2d(output_dir, mask_path_close)