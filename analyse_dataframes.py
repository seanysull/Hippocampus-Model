# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:27:52 2020

@author: seano
"""
import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import os
from matplotlib import cm
from scipy.ndimage import gaussian_filter, label, generate_binary_structure
# =============================================================================
# import seaborn as sns
# =============================================================================


def create_dataframes(apt=0.96):
    active_pixels_threshold=apt
    num_bins = 25*50
    min_pf_size = 6
    max_pf_size = 200
# =============================================================================
#     min_pf_size = round(num_bins*0.01)
#     max_pf_size = round(num_bins*0.2)
# =============================================================================
# =============================================================================
#     dg = np.load("rate_maps/numpy/simulation_data_2807_50000steps.h5_DG.npy")
#     ca3 = np.load("rate_maps/numpy/simulation_data_2807_50000steps.h5_CA3.npy")
#     ca1 = np.load("rate_maps/numpy/simulation_data_2807_50000steps.h5_CA1.npy")
# =============================================================================
    
# =============================================================================
#     npy_files = os.listdir("rate_maps/numpy")
# =============================================================================
    
    s = generate_binary_structure(2,2)
    dataframe = pd.DataFrame(columns=["dataset","layer","node_number","pf_count",
                                      "pf1_size","pf2_size","pf3_size","pf4_size",
                                      "pf5_size","pf6_size"])
    npy_files= ["simulation_data_1608_100000steps_stretched_X.h5_CA1_50bin.npy",
                "simulation_data_1608_100000steps_stretched_X.h5_CA3_50bin.npy",
                "simulation_data_1608_100000steps_stretched_X.h5_DG_50bin.npy",
                "simulation_data_2807_50000steps.h5_base_CA1.npy",
                "simulation_data_2807_50000steps.h5_base_CA3.npy",
                "simulation_data_2807_50000steps.h5_base_DG.npy"]
    
    for file in npy_files:
        maps = np.load("rate_maps/numpy/"+file)                         
        if "stretched_X" in file:
            dset = "stx"
        elif "morphed25" in file:
            dset = "m25"
        elif "morphed50" in file:
            dset = "m50"
        elif "morphed75" in file:
            dset = "m75"
        elif "morphed100" in file:
            dset = "m100"
        else:
            dset = "base" 
        if "CA1" in file:
            layer = "CA1"
        elif "CA3" in file:
            layer = "CA3"
        else:
            layer = "DG"
            
        if dset[-1] == "_":
            dset = dset[:-1]
        for node_number, ratemap in enumerate(maps):
            row_to_add = {"dataset":dset,"layer":layer,"node_number":node_number,
                          "pf_count":0, "pf1_size":np.nan,"pf2_size":np.nan,
                          "pf3_size":np.nan,"pf4_size":np.nan,"pf5_size":np.nan,
                          "pf6_size":np.nan}
            
            ratemap[ ratemap <  ratemap.max()*active_pixels_threshold ] = 0
            ratemap[ ratemap >= ratemap.max()*active_pixels_threshold ] = 1
            labeled_array, num_features = label(ratemap, structure=s)
            sizes=[]
            for group in range(1,num_features+1):
                size = np.sum(labeled_array==group)
                if size <= max_pf_size and size >= min_pf_size:
                    sizes.append(size)
                   
            num_pfs = len(sizes)
            if num_pfs>0:
                row_to_add["pf_count"]=num_pfs
                sizes.sort(reverse=True) 
                for ind, size in enumerate(sizes):
                    row_to_add["pf"+str(ind+1)+"_size"] = size
            
            dataframe = dataframe.append(row_to_add,ignore_index=True)
        
    dataframe.to_pickle("2550bins_stretch_96thresh.pkl")
    
def plot_count_size():
    frame_paths = os.listdir("rate_maps/dataframes")
    df_names = ["95","96","97"]
    fig, axes = plt.subplots(nrows=1,ncols=3,sharey=True)
    fig.suptitle("Number of place fields per cell per layer")
    fig.text(0.5, 0.04, '# of place-fields', ha='center', va='center')
    fig.text(0.06, 0.5, '# of cells (%)', ha='center', va='center', rotation='vertical')
    col = 2
    width=0.9
    height=5

    for df_name,frame_path in zip(df_names,frame_paths):
        data = pd.read_pickle("rate_maps/dataframes/"+frame_path)
        base = data[data["dataset"]=="base"]
        groups = base.groupby(["layer"])
        col = 2
        x = np.arange(6)  # the label locations
        for ind,group in groups:            
            ax = axes[col]
            counts = group["pf_count"].value_counts().reindex(data.pf_count.unique(), fill_value=0)
            counts = counts.sort_index().to_numpy()
            fractional = counts/counts.sum()
            ax.set_title(ind)
# =============================================================================
#             ax.grid()
# =============================================================================
            ax.set_xticks([0,1,2,3,4])
            
            if df_name=="95":
                pass
# =============================================================================
#                 ax.bar(x-width-0.05, fractional, width=width, 
#                        bottom=None, align='center',label=df_name)
# =============================================================================
                
            elif df_name=="96":
                ax.bar(x, fractional, width=width, 
                       bottom=None, align='center')
                
            elif df_name=="97":
                pass
# =============================================================================
#                 ax.bar(x+width+0.05, fractional, width=width, 
#                        bottom=None, align='center',label=df_name)
# =============================================================================
            
            col-=1
        fig.savefig("Place_field_count_histogram")   

        
def plot_average_sizes():
    frame_paths = os.listdir("rate_maps/dataframes")
    df_names = ["95","96","97"]
    fig, axes = plt.subplots(nrows=1,ncols=3,sharey=True)
    fig.suptitle("Number of place fields per cell per layer")
    fig.text(0.5, 0.04, '# of place-fields', ha='center', va='center')
    fig.text(0.06, 0.5, '# of cells (%)', ha='center', va='center', rotation='vertical')
    col = 2
    width=0.25
    height=5

    for df_name,frame_path in zip(df_names,frame_paths):
        data = pd.read_pickle("rate_maps/dataframes/"+frame_path)
        base = data[data["dataset"]=="base"]
        groups = base.groupby(["layer"])
        col = 2
        x = np.arange(5)  # the label locations
        for ind,group in groups:  
            avg_sizes = group[["pf1_size","pf2_size","pf3_size","pf4_size","pf5_size"]].mean()
            avg_sizes = avg_sizes.fillna(value=0).to_numpy()
            
            ax = axes[col]

            ax.set_title(ind)
            ax.grid()
            
            if df_name=="95":
                pass
# =============================================================================
#                 ax.bar(x-width-0.05, avg_sizes, width=width, 
#                        bottom=None, align='center',label=df_name)
# =============================================================================
                
            elif df_name=="96":
                ax.bar(x, avg_sizes, width=width, 
                       bottom=None, align='center',label=df_name)
                
            elif df_name=="97":
                pass
# =============================================================================
#                 ax.bar(x+width+0.05, avg_sizes, width=width, 
#                        bottom=None, align='center',label=df_name)
# =============================================================================
# =============================================================================
#             ax.legend()
# =============================================================================
            col-=1    

def plot_receptive_field():

    ca1 = np.load("rate_maps/numpy/simulation_data_2807_50000steps.h5_CA1.npy")
    ca3 = np.load("rate_maps/numpy/simulation_data_2807_50000steps.h5_CA3.npy")
    dg = np.load("rate_maps/numpy/simulation_data_2807_50000steps.h5_DG.npy")
    names = ["dg","ca3","ca1"]
    dg_inds = [1,4,16,19,36,86,87,33,149]
    ca3_inds = [3,24,26,27,37,38,43,64,65]
    ca1_inds = [47,59,61,62,64,76,88,104,107]
    fig, axes = plt.subplots(3,9)
    
# =============================================================================
#     for name,maps in zip(names,[dg,ca3,ca1]):
# =============================================================================
        
    for cell in range(9):

        dg_cell = dg[dg_inds[cell],:,:].copy()
        dg_cell[dg_cell < dg_cell.max()*0.96] = 0  
        dg_cell = gaussian_filter(dg_cell,sigma=0.75)
        
        ca3_cell = ca3[ca3_inds[cell],:,:].copy()
        ca3_cell[ca3_cell < ca3_cell.max()*0.96] = 0  
        ca3_cell = gaussian_filter(ca3_cell,sigma=0.75)
        
        ca1_cell = ca1[ca1_inds[cell],:,:].copy()
        ca1_cell[ca1_cell < ca1_cell.max()*0.96] = 0  
        ca1_cell = gaussian_filter(ca1_cell,sigma=0.75)        
                
        ax_dg = axes[0,cell]
        ax_ca3 = axes[1,cell]
        ax_ca1 = axes[2,cell]
        
        ax_dg.set_title("DG "+str(dg_inds[cell]))
        ax_dg.grid(False)
        ax_dg.imshow(dg_cell, interpolation="bilinear", cmap=cm.afmhot)
        ax_dg.axis("off")

        ax_ca3.set_title("CA3 "+str(ca3_inds[cell]))
        ax_ca3.grid(False)
        ax_ca3.imshow(ca3_cell, interpolation="bilinear", cmap=cm.afmhot)
        ax_ca3.axis("off")

        ax_ca1.set_title("CA1 "+str(ca1_inds[cell]))
        ax_ca1.grid(False)
        ax_ca1.imshow(ca1_cell, interpolation="bilinear", cmap=cm.afmhot)
        ax_ca1.axis("off")
        
    plt.tight_layout()
    fig.savefig('rate_maps/big_fella.png', format='png', dpi=600)

def place_field_size_modification():
    data = pd.read_pickle("rate_maps/dataframes/25bins_96thresh.pkl")
    data = data[data["dataset"].isin(["base","m25","m50","m75","m100"])]
    groups = data.groupby(["dataset","layer"])
    avg = pd.DataFrame(index=["base","m25","m50","m75","m100"],columns=["DG","CA3","CA1"])
    for ind,group in groups:
        group = group[["pf1_size","pf2_size","pf3_size","pf4_size"]]
        means = group.mean()
        num_pc = group.count()
        weights = num_pc/num_pc.sum()
        wm = means*weights
        final = wm.sum()
        avg.loc[ind] = final

# =============================================================================
#     for ind,group in groups:
#         group = group["pf1_size"]
#         mean = group.mean()
#         avg.loc[ind] = mean
#         print(ind)
# =============================================================================
    nump = avg.to_numpy()
    fig, ax = plt.subplots()
    ax.plot([0,1,2,3,4],nump)
    start, end = 0,5
    ax.xaxis.set_ticks(np.arange(start, end, 1))
    ax.xaxis.set_ticklabels(["0%","25%","50%","75%","100%"])
    ax.set_ylabel("Place field size in bins")
    ax.set_xlabel("Environment modification level")
    ax.set_title("Place fields under environment modification")
    fig.savefig("plots/pf_change_envmod")

    a=1            
                
def population_vector_correlation():
    base_dg = np.load("rate_maps/numpy/simulation_data_2807_50000steps.h5_base_DG.npy").reshape(-1,625)
    base_ca3 = np.load("rate_maps/numpy/simulation_data_2807_50000steps.h5_base_CA3.npy").reshape(-1,625)
    base_ca1 = np.load("rate_maps/numpy/simulation_data_2807_50000steps.h5_base_CA1.npy").reshape(-1,625)

    m25_dg = np.load("rate_maps/numpy/simulation_data_2708_50000steps_morphed25.h5_DG.npy").reshape(-1,625)
    m25_ca3 = np.load("rate_maps/numpy/simulation_data_2708_50000steps_morphed25.h5_CA3.npy").reshape(-1,625)
    m25_ca1 = np.load("rate_maps/numpy/simulation_data_2708_50000steps_morphed25.h5_CA1.npy").reshape(-1,625)
    
    m50_dg = np.load("rate_maps/numpy/simulation_data_2708_50000steps_morphed50.h5_DG.npy").reshape(-1,625)
    m50_ca3 = np.load("rate_maps/numpy/simulation_data_2708_50000steps_morphed50.h5_CA3.npy").reshape(-1,625)
    m50_ca1 = np.load("rate_maps/numpy/simulation_data_2708_50000steps_morphed50.h5_CA1.npy").reshape(-1,625)
    
    m75_dg = np.load("rate_maps/numpy/simulation_data_3108_50000steps_morphed75.h5_DG.npy").reshape(-1,625)
    m75_ca3 = np.load("rate_maps/numpy/simulation_data_3108_50000steps_morphed75.h5_CA3.npy").reshape(-1,625)
    m75_ca1 = np.load("rate_maps/numpy/simulation_data_3108_50000steps_morphed75.h5_CA1.npy").reshape(-1,625)
    
    m100_dg = np.load("rate_maps/numpy/simulation_data_3108_50000steps_morphed100.h5_DG.npy").reshape(-1,625)
    m100_ca3 = np.load("rate_maps/numpy/simulation_data_3108_50000steps_morphed100.h5_CA3.npy").reshape(-1,625)
    m100_ca1 = np.load("rate_maps/numpy/simulation_data_3108_50000steps_morphed100.h5_CA1.npy").reshape(-1,625)
    
    avgs_dg=[]
    for morph in [base_dg, m25_dg,m50_dg,m75_dg,m100_dg]:
        coeffs=[]
        for ind,(x,y) in enumerate(zip(base_dg.T,morph.T)):
            corr = np.corrcoef(x,y)[0][1]
            coeffs.append(corr)
         
        avg = np.mean(coeffs)
        avgs_dg.append(avg)
    
    avgs_ca3=[]
    for morph in [base_ca3, m25_ca3,m50_ca3,m75_ca3,m100_ca3]:
        coeffs=[]
        for ind,(x,y) in enumerate(zip(base_ca3.T,morph.T)):
            corr = np.corrcoef(x,y)[0][1]
            coeffs.append(corr)
         
        avg = np.mean(coeffs)
        avgs_ca3.append(avg)
        
    avgs_ca1=[]
    for morph in [base_ca1, m25_ca1,m50_ca1,m75_ca1,m100_ca1]:
        coeffs=[]
        for ind,(x,y) in enumerate(zip(base_ca1.T,morph.T)):
            corr = np.corrcoef(x,y)[0][1]
            coeffs.append(corr)
         
        avg = np.mean(coeffs)
        avgs_ca1.append(avg)
    
    x = [0,1,2,3]
    neting = [avgs_dg[1],avgs_dg[2],avgs_dg[4],avgs_dg[3]]
    neting_2 = [avgs_ca3[1],avgs_ca3[2],avgs_ca3[4],avgs_ca3[3]]
    neting_3 = [avgs_ca1[1],avgs_ca1[2],avgs_ca1[4],avgs_ca1[3]]
    fig, ax = plt.subplots()
    ax.plot(neting_2)
    ax.plot(neting)
    ax.plot(neting_3)
    ax.xaxis.set_ticks(np.arange(0, 4, 1))
    ax.xaxis.set_ticklabels(["0-25%","25-50%","50-75%","75-100%"])
    ax.set_ylabel("PV correlation")
    ax.set_xlabel("Environment modification level")
    ax.set_title("Rate modulation in DG, CA3, CA1")
    ax.legend(labels=["dg","ca3","ca1"])
    fig.savefig("plots/pv_corr_ratemod_alllayer")
    

def stretched_rate_maps():
    base_ca1 = np.load("rate_maps/numpy/simulation_data_2807_50000steps.h5_base_CA1.npy")
    base_ca3 = np.load("rate_maps/numpy/simulation_data_2807_50000steps.h5_base_CA3.npy")
    base_dg = np.load("rate_maps/numpy/simulation_data_2807_50000steps.h5_base_DG.npy")

    stretch_ca1 = np.load("rate_maps/numpy/simulation_data_1608_100000steps_stretched_X.h5_CA1_50bin.npy")
    stretch_ca3 = np.load("rate_maps/numpy/simulation_data_1608_100000steps_stretched_X.h5_CA3_50bin.npy")
    stretch_dg = np.load("rate_maps/numpy/simulation_data_1608_100000steps_stretched_X.h5_DG_50bin.npy")
    
    names = ["dg","ca3","ca1"]
    dg_inds = [1,4,16,19,36,86,87,33,149]
    ca3_inds = [3,24,26,27,37,38,43,64,65]
    ca1_inds = [47,59,61,62,64,76,88,104,107]

    
# =============================================================================
#     for name,maps in zip(names,[dg,ca3,ca1]):
# =============================================================================
        
    for cell in range(160):
        fig, axes = plt.subplots(2,1)
        
# =============================================================================
#         fig_1, axes_1 = plt.subplots()
#         fig_2, axes_3 = plt.subplots()
# =============================================================================
# =============================================================================
#         dg_cell = base_dg[dg_inds[cell],:,:].copy()
# =============================================================================
        dg_cell = base_ca1[cell,:,:].copy()
        dg_cell[dg_cell < dg_cell.max()*0.96] = 0  
        dg_cell = gaussian_filter(dg_cell,sigma=0.75)
        
# =============================================================================
#         stretch_dg_cell = stretch_dg[ca3_inds[cell],:,:].copy()
# =============================================================================
        stretch_dg_cell = stretch_ca1[cell,:,:].copy()
        stretch_dg_cell[stretch_dg_cell < stretch_dg_cell.max()*0.96] = 0  
        stretch_dg_cell = gaussian_filter(stretch_dg_cell,sigma=0.75)
        
              
        ax_dg = axes[0]
        ax_ca3 = axes[1]
        
        ax_dg.set_title("Base "+str(cell))
        ax_dg.grid(False)
        ax_dg.imshow(dg_cell, interpolation="bilinear", cmap=cm.afmhot)
        ax_dg.axis("off")
        ax_dg.set_aspect(1,anchor="SW")

        ax_ca3.set_title("Stretch "+str(cell))
        ax_ca3.grid(False)
        ax_ca3.imshow(stretch_dg_cell, interpolation="bilinear", cmap=cm.afmhot)
        ax_ca3.axis("off")
        ax_ca3.set_aspect(1,anchor="SW")
        

        
        plt.tight_layout()
        fig.savefig('rate_maps/CA1_stretched/'+str(cell)+'.png', format='png', dpi=600)           
        plt.close()

def stretched_count_diff():
    data = pd.read_pickle("2550bins_stretch_96thresh.pkl")     
    stretch_dg = data[(data["dataset"]=="stx")& (data["layer"]=="DG")].reset_index(drop=True)
    stretch_ca3 = data[(data["dataset"]=="stx")& (data["layer"]=="CA3")].reset_index(drop=True)
    stretch_ca1 = data[(data["dataset"]=="stx")& (data["layer"]=="CA1")].reset_index(drop=True)
    
    data = pd.read_pickle("rate_maps/dataframes/25bins_96thresh.pkl")
    base_dg = data[(data["dataset"]=="base")& (data["layer"]=="DG")].reset_index(drop=True)
    base_ca3 = data[(data["dataset"]=="base")& (data["layer"]=="CA3")].reset_index(drop=True)
    base_ca1 = data[(data["dataset"]=="base")& (data["layer"]=="CA1")].reset_index(drop=True)
    
    pf_count_diff_dg = stretch_dg["pf_count"]-base_dg["pf_count"]
    pf_count_diff_ca3 = stretch_ca3["pf_count"]-base_ca3["pf_count"]
    pf_count_diff_ca1 = stretch_ca1["pf_count"]-base_ca1["pf_count"]
    counts = [pf_count_diff_dg, pf_count_diff_ca3, pf_count_diff_ca1]

    fig, axes = plt.subplots(nrows=3,ncols=1,sharex=True, sharey=True)
    # Set common labels
    fig.text(0.5, 0.04, 'Number of place-fields', ha='center', va='center')
    fig.text(0.06, 0.5, 'Layer', ha='center', va='center', rotation='vertical')
    fig.suptitle("Change in number of place-fields after stretching.")
    labels=["DG","CA3","CA1"]
    for ax, data, label in zip(axes,counts,labels):
        ax.violinplot(data, vert=False, showmeans=True,showmedians=False)
        ax.set_ylabel(label,rotation="horizontal",ha="right")
        ax.set_yticks([])
    fig.savefig("stretching_number_pfields")  

def stretched_size_diff():
    data = pd.read_pickle("2550bins_stretch_96thresh.pkl")
    sizes = data.melt(id_vars=["dataset","layer"],value_vars=["pf1_size","pf2_size",
                                                             "pf3_size","pf4_size",
                                                             "pf5_size"]).dropna()
# =============================================================================
#     stretch_dg = sizes[(sizes["dataset"]=="stx")& (sizes["layer"]=="DG")].reset_index(drop=True)
#     stretch_ca3 = sizes[(sizes["dataset"]=="stx")& (sizes["layer"]=="CA3")].reset_index(drop=True)
#     stretch_ca1 = sizes[(sizes["dataset"]=="stx")& (sizes["layer"]=="CA1")].reset_index(drop=True)
# =============================================================================
    sizes = sizes[sizes["dataset"]=="stx"]
    data_2= pd.read_pickle("rate_maps/dataframes/25bins_96thresh.pkl")
    sizes_base = data_2.melt(id_vars=["dataset","layer"],value_vars=["pf1_size","pf2_size",
                                                             "pf3_size","pf4_size",
                                                             "pf5_size"]).dropna()    
# =============================================================================
#     base_dg = sizes[(sizes["dataset"]=="base")& (sizes["layer"]=="DG")].reset_index(drop=True)
#     base_ca3 = sizes[(sizes["dataset"]=="base")& (sizes["layer"]=="CA3")].reset_index(drop=True)
#     base_ca1 = sizes[(sizes["dataset"]=="base")& (sizes["layer"]=="CA1")].reset_index(drop=True)
# =============================================================================
    sizes_base = sizes_base[sizes_base["dataset"]=="base"]
# =============================================================================
#     counts = [(base_dg,stretch_dg), (base_ca3,stretch_ca1), (base_ca1,stretch_ca1)]
# =============================================================================

    all_data = pd.concat([sizes,sizes_base])
    g = sns.FacetGrid(all_data, row="layer",hue="dataset", height=2, aspect=3, legend_out = False)
    plt.xlim([0,250])
    g.map(sns.distplot, "value", hist=False,  kde_kws={'clip': (6,300)}, rug=True)
    g.add_legend()
    new_labels = ['Stretched', 'Base']
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
# =============================================================================
#     g.fig.suptitle("Place field size distribution")
# =============================================================================
    g.set_axis_labels("Size of PF", "Fraction of Cells")
    plt.savefig("plots/pf_size_stretching_2.png")

   


def error_spatial_map_morphing():
    arrays = np.load("spatial_mse.npy")
    fig, axes = plt.subplots(1,5, sharey=True, )
    low = arrays.min()
    high = arrays.max()    
    low_rect_25 = mpl.patches.Rectangle((0, -0.05), width=1, height=0.025, 
                                 color="green", transform=axes[1].transAxes, clip_on=False)

    low_rect_50 = mpl.patches.Rectangle((0, -0.05), width=1, height=0.025, 
                                 color="green", transform=axes[2].transAxes, clip_on=False)

    low_rect_75 = mpl.patches.Rectangle((0, -0.05), width=1, height=0.025, 
                                 color="green", transform=axes[3].transAxes, clip_on=False)

    low_rect_100 = mpl.patches.Rectangle((0, -0.05), width=1, height=0.025, 
                                 color="green", transform=axes[4].transAxes, clip_on=False)

    right_rect_50 = mpl.patches.Rectangle((1.025,0 ), width=0.025, height=1, 
                                 color="green", transform=axes[2].transAxes, clip_on=False)

    right_rect_75 = mpl.patches.Rectangle((1.025,0 ), width=0.025, height=1, 
                                 color="green", transform=axes[3].transAxes, clip_on=False)

    right_rect_100 = mpl.patches.Rectangle((1.025,0 ), width=0.025, height=1, 
                                 color="green", transform=axes[4].transAxes, clip_on=False)
    
    high_rect_75 = mpl.patches.Rectangle((0, 1.025), width=1, height=0.025, 
                                 color="green", transform=axes[3].transAxes, clip_on=False)

    high_rect_100 = mpl.patches.Rectangle((0, 1.025), width=1, height=0.025, 
                                 color="green", transform=axes[4].transAxes, clip_on=False)
    
    left_rect = mpl.patches.Rectangle((-0.05, 0), width=0.025, height=1, 
                                 color="green", transform=axes[4].transAxes, clip_on=False)    
    axes[0].imshow(arrays[0], cmap='hot', interpolation='nearest', vmin=low, vmax=high)
    axes[0].set_title("0% ")
    axes[0].xaxis.set_visible(False)
    axes[0].yaxis.set_visible(False)
    
    axes[1].imshow(arrays[1], cmap='hot', interpolation='nearest', vmin=low, vmax=high)
    axes[1].add_patch(low_rect_25)
    axes[1].set_title("25% ")
    axes[1].xaxis.set_visible(False)
    axes[1].yaxis.set_visible(False)
    
    axes[2].imshow(arrays[2], cmap='hot', interpolation='nearest', vmin=low, vmax=high)
    axes[2].add_patch(low_rect_50)
    axes[2].add_patch(right_rect_50)
    axes[2].set_title("50% ")
    axes[2].xaxis.set_visible(False)
    axes[2].yaxis.set_visible(False)

    axes[3].imshow(arrays[3], cmap='hot', interpolation='nearest', vmin=low, vmax=high)
    axes[3].add_patch(low_rect_75)
    axes[3].add_patch(right_rect_75)
    axes[3].add_patch(high_rect_75)
    axes[3].set_title("75% ")
    axes[3].xaxis.set_visible(False)
    axes[3].yaxis.set_visible(False)
    
    im = axes[4].imshow(arrays[4], cmap='hot', interpolation='nearest', vmin=low, vmax=high)
    axes[4].add_patch(low_rect_100)
    axes[4].add_patch(right_rect_100)
    axes[4].add_patch(high_rect_100)
    axes[4].add_patch(left_rect)
    axes[4].set_title("100% ")
    axes[4].xaxis.set_visible(False)
    axes[4].yaxis.set_visible(False)
    
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.01, pad=0.04)
    fig.savefig("plots/morphing_spatial_mse", bbox_inches="tight")
# =============================================================================
#     plt.tight_layout()
# =============================================================================

def plot_hippo_losses():
    base = np.load("hippo_base_train_losses_1000.npy")
    m25 = np.load("data/simulation_data_2708_50000steps.h5_denoiseV4_morph25_embeddings.h5.npy")
    m50 = np.load("data/simulation_data_2708_50000steps.h5_denoiseV4_morphed50_embeddings.h5.npy")
    m75 = np.load("data/simulation_data_3108_50000steps.h5_denoiseV4_morphed75_embeddings.h5.npy")
    m100 = np.load("data/simulation_data_3108_50000steps.h5_denoiseV4_morphed100_embeddings.h5.npy")
    base_m = base -0.001
    fig, ax = plt.subplots(1,1)
# =============================================================================
#     ax.ticklabel_format(axis="y",style="sci")
# =============================================================================
    ax.plot(base[200:],label="0%")
    ax.plot(m25,label="25%")
    ax.plot(m50,label="50%")
    ax.plot(m75,label="75%")
    ax.plot(m100,label="100%")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.legend()
    ax.set_title("Relearning in modified environment")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.savefig("relearning")

def count_sequences(sequence):
    counts = []
    in_seq = False
    index=-1
    for ind, val in enumerate(sequence):
        if val > 0:
            if in_seq == False:
                in_seq = True
                counts.append(1)
                index +=1
            else:
                counts[index]+=1
        else:
            in_seq = False
                      
    
    counts_deg = [x/2 for x in counts]
    counts_final = [x for x in counts if x>=45]
    return counts_final     
        
        
def plot_circ_ratemaps():
    num_bins = 720
    bin_edges = np.arange(0,360,0.5)
    theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / num_bins)
    dg = np.load("rate_maps/numpy/DG_orientation_base_bins720.npy")
    count=0
    for cell in dg:
        blurred = gaussian_filter(cell,sigma=5).copy()
        blurred[blurred<blurred.max()*0.9] = 0
# =============================================================================
#         cell[cell<cell.max()*0.9] = 0
# =============================================================================
        reblurred = gaussian_filter(blurred,sigma=5)
        fig = plt.figure() 
# =============================================================================
#         ax = plt.subplot(111, projection='polar')
# =============================================================================
        ax = plt.subplot(111)
        ax.plot(reblurred, color="purple")
# =============================================================================
#         ax.plot(theta, reblurred, color="purple")
# =============================================================================

        ax.grid(True)
        
        ax.set_title("A line plot on a polar axis", va='bottom')
        fig.savefig("rate_maps/DG_Orient/dg_top10"+str(count))
        plt.close(plt.gcf())
        count+=1

def get_orientation_fields():
    dg = np.load("rate_maps/numpy/DG_orientation_base_bins720.npy")
    counts = []
    for cell in dg:
        blurred = gaussian_filter(cell,sigma=5).copy()
        blurred[blurred<blurred.max()*0.9] = 0
        reblurred = gaussian_filter(blurred,sigma=5)
        count = count_sequences(reblurred)
        counts.append(count)
    a=1
    
        

    
if __name__ == "__main__":
    
# =============================================================================
#     plot_count_size()
# =============================================================================
# =============================================================================
#     plot_average_sizes()
# =============================================================================
# =============================================================================
#     plot_receptive_field()
# =============================================================================
# =============================================================================
#     place_field_size_modification()
# =============================================================================
    population_vector_correlation()
# =============================================================================
#     stretched_rate_maps()
# =============================================================================
# =============================================================================
#     inspect_rate_maps()
# =============================================================================
# =============================================================================
#     stretched_count_diff()
# =============================================================================
# =============================================================================
#     stretched_size_diff()
# =============================================================================
# =============================================================================
#     circ_analysis()
# =============================================================================
# =============================================================================
#     error_with_morphing()
# =============================================================================
# =============================================================================
#     error_spatial_map_morphing()
# =============================================================================
# =============================================================================
#     plot_hippo_losses()
# =============================================================================
# =============================================================================
#     get_orientation_fields()
# =============================================================================
