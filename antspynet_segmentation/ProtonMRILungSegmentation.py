#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:04:14 2019

@author: ftan1
"""
import os
import sys
import time
import numpy as np
import keras

import ants
import antspynet

def ProtonMRILungSegmentation(input_file_name, output_file_name_prefix, reorient_template_file_name):
#    if len(args) != 4:
#        help_message = ("Usage:  python doLungSegementation.py" +
#            " inputFile outputFilePrefix reorientationTemplate")
#        raise AttributeError(help_message)
#    else:
#        input_file_name = args[1]
#        output_file_name_prefix = args[2]
#        reorient_template_file_name = args[3]
    
    classes = ("Background", "LeftLung", "RightLung")
    number_of_classification_labels = len(classes)
    
    image_mods = ["Proton"]
    channel_size = len(image_mods)
    
    print("Reading reorientation template " + reorient_template_file_name)
    start_time = time.time()
    reorient_template = ants.image_read(reorient_template_file_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("  (elapsed time: ", elapsed_time, " seconds)")
    
    resampled_image_size = reorient_template.shape
    
    unet_model = antspynet.create_unet_model_3d((*resampled_image_size, channel_size),
      number_of_outputs = number_of_classification_labels,
      number_of_layers = 4,
      number_of_filters_at_base_layer = 16,
      dropout_rate = 0.0,
      convolution_kernel_size = (7, 7, 5),
      deconvolution_kernel_size = (7, 7, 5))
    
    print( "Loading weights file" )
    start_time = time.time()
    weights_file_name = "./lungSegmentationWeights.h5"
    
    if not os.path.exists(weights_file_name):
        weights_file_name = antspynet.get_pretrained_network("protonLungMri", weights_file_name)
    
    unet_model.load_weights(weights_file_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("  (elapsed time: ", elapsed_time, " seconds)")
    
    start_time_total = time.time()
    
    print( "Reading ", input_file_name )
    start_time = time.time()
    image = ants.image_read(input_file_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("  (elapsed time: ", elapsed_time, " seconds)")
    
    print( "Normalizing to template" )
    start_time = time.time()
    center_of_mass_template = ants.get_center_of_mass(reorient_template)
    center_of_mass_image = ants.get_center_of_mass(image)
    translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
      center=np.asarray(center_of_mass_template),
      translation=translation)
    warped_image = ants.apply_ants_transform_to_image(xfrm, image,
      reorient_template, interpolation='linear')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("  (elapsed time: ", elapsed_time, " seconds)")
    
    batchX = np.expand_dims(warped_image.numpy(), axis=0)
    batchX = np.expand_dims(batchX, axis=-1)
    batchX = (batchX - batchX.mean()) / batchX.std()
    
    print("Prediction and decoding")
    start_time = time.time()
    predicted_data = unet_model.predict(batchX, verbose=0)
    
    origin = warped_image.origin
    spacing = warped_image.spacing
    direction = warped_image.direction
    
    probability_images_array = list()
    for i in range(number_of_classification_labels):
        probability_images_array.append(
           ants.from_numpy(np.squeeze(predicted_data[0, :, :, :, i]),
             origin=origin, spacing=spacing, direction=direction))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("  (elapsed time: ", elapsed_time, " seconds)")
    
    print("Renormalize to native space")
    start_time = time.time()
    for i in range(number_of_classification_labels):
        probability_images_array[i] = ants.apply_ants_transform_to_image(
          ants.invert_ants_transform(xfrm), probability_images_array[i], image)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("  (elapsed time: ", elapsed_time, " seconds)")
    
    for i in range(1, number_of_classification_labels):
        print("Writing", classes[i])
        start_time = time.time()
        ants.image_write(probability_images_array[i],
          output_file_name_prefix + classes[i] + "Probability.nii.gz")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("  (elapsed time: ", elapsed_time, " seconds)")
    
    end_time_total = time.time()
    elapsed_time_total = end_time_total - start_time_total
    print("Total elapsed time: ", elapsed_time_total, "seconds")