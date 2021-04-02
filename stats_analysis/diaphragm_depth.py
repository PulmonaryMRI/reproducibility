#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:32:53 2021

@author: ftan1
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

nvol = 6
tb_nrows = nvol * 4
nphase = 12
nscan = 2
table = pd.read_excel('/home/ftan1/Downloads/Tables.xls', usecols=("D:O"), nrows = tb_nrows)
df = pd.DataFrame()
df['diaphragm position'] = table.values.flatten()
df['volunteer'] = np.int8(np.floor(np.linspace(0,nvol,nvol*nphase*4, endpoint=False)) + 1) 
df['scan'] = np.int8(np.tile(np.floor(np.linspace(0,nscan,nphase*4, endpoint=False)) + 1, nvol))
df['lung position'] = (['Left'] * nphase + ['Right'] * nphase) * nscan * nvol
df['phase'] = np.int8(np.tile(np.linspace(1,nphase,nphase), nvol * 4))
output_dir = '/data/larson4/UTE_Lung/2021-03-12_vo/reg/P86528/3DnT_BSpline/'

# fig, axes = plt.subplots(1,5, figsize=(24, 4), sharey=True)

# for iter in range(5):
#     df_vol = df[df["volunteer"]==(iter+1)]
    
#     if iter < 4 :
#         g = sns.lineplot(ax=axes[iter], x = "phase", y ="diaphragm position", hue="scan", style="lung position", palette="deep", ci=None, markers=True, legend=False, data = df_vol)
#         g.set(xticks =np.int8(np.linspace(1,12,12)), ylabel = "diaphragm position (mm)")
#     else:
#         g=sns.lineplot(ax=axes[iter], x = "phase", y ="diaphragm position", hue="scan", style="lung position", palette="deep", ci=None, markers=True, legend="full", data = df_vol)
#         g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#         g.set(xticks =np.int8(np.linspace(1,12,12)))      
# plt.savefig('/home/ftan1/Downloads/diaphragm_position.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})
g = sns.relplot(data = df, x = "phase", y ="diaphragm position", hue="scan", style="lung position", row = "volunteer", palette="deep", ci=None, markers=True, legend=False,  kind = "line", aspect = 1.2)
g.set(xticks = [1,2,3,4,5,6,7,8,9,10,11,12])
g.savefig(output_dir + 'diaphragm_position.png', bbox_inches='tight', dpi = 300)