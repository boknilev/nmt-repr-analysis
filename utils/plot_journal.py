# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:10:01 2017

@author: belinkov
"""

import sys
assert sys.version_info[0] == 3, 'Must run with Python 3'
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
#plt.style.use('ggplot')
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')

# put all result files here
RESULTS_DIR = '../results'

langs = ['ar', 'es', 'fr', 'ru', 'zh', 'en']
pairs = [['en', 'ar'], ['en', 'es'], ['en', 'fr'], ['en', 'ru'], ['en', 'zh'], 
         ['en', 'en'],
         ['ar', 'en'], ['es', 'en'], ['fr', 'en'], ['ru', 'en'], ['zh', 'en']]
pairs_en = [['en', 'ar'], ['en', 'es'], ['en', 'fr'], ['en', 'ru'], ['en', 'zh'], 
         ['en', 'en']]
pretty_lang_names = {'en': 'English', 'ar':'Arabic', 'es':'Spanish', 'fr':'French', 'es':'Spanish', 'ru':'Russian', 'zh':'Chinese'}
pretty_dist_names = {'dist1': '1', 'dist2': '2', 'dist3': '3', 'dist4': '4', 'dist5': '5', 'dist6-7': '6-7', 'dist8-10': '8-10', 'dist10-': '>10' }
pretty_dist_names_list = ('1', '2', '3', '4', '5', '6-7', '8-10', '>10')

en_maj, en_mfl = 0.1332812025*100, 0.4047091533*100
ar_maj, ar_mfl = 0.2256017981*100, 0.5023926914*100
es_maj, es_mfl = 0.1499913599*100, 0.4269051322*100
fr_maj, fr_mfl = 0.1334860475*100, 0.4328404831*100
ru_maj, ru_mfl = 0.1876313145*100, 0.3099479309*100
zh_maj, zh_mfl = 0.1468902015*100, 0.3564107019*100
majs = [en_maj]*6 + [ar_maj, es_maj, fr_maj, ru_maj, zh_maj]
mfls = [en_mfl]*6 + [ar_mfl, es_mfl, fr_mfl, ru_mfl, zh_mfl]


def get_accs_from_df(df, col_pref='acc'):    
    
    accs = [df[col].values for col in df.columns if col.startswith(col_pref)]
    return accs    
    

def load_data(filename, sep='\t'):
    
    df = pd.read_csv(filename, sep=sep)
    df['mean'] = np.mean([df['acc1'], df['acc2'], df['acc3']], axis=0)
    df['std'] = np.std([df['acc1'], df['acc2'], df['acc3']], axis=0)
    return df


def load_data_by_distance(filename, sep='\t', scale=100):
    
    df = pd.read_csv(filename, sep=sep)
    dists = [col for col in df.columns if col.startswith('dist')]
    for dist in dists:
        df[dist] *= scale
#    for source in df.source.unique():
#        for target in df.target.unique():
#            for layer in df.layer.unique():
#                df_source_target_layer = df[(df['source'] == source) & (df['target'] == target) & (df['layer'] == layer)]
#                df_mean = df_source_target_layer.mean()
#                df_std = df_source_target_layer.std()
#                mean_dists = [df_mean[d] for d in dists]
#                std_dists = [df_std[d] for d in dists]
#                series_mean = pd.Series([layer, source, target, 'mean'] + mean_dists, index=df.columns)
#                series_std = pd.Series([layer, source, target, 'std'] + std_dists, index=df.columns)
#                df.append(series_mean, ignore_index=True)
#                df.append(series_std, ignore_index=True)
    return df
    

def load_data_by_type(filename, sep='\t', scale=100):
    
    df = pd.read_csv(filename, sep=sep)
    types = df.columns[4:]
    for t in types:
        df[t] = df[t]
        df[t] *= scale
    return df


def load_data_by_type_all(filename, sep='\t', scale=100):
    """ Load data for all language pairs by type

    Convert types from column to values of a "relation" column
    Empty cells are possible and they are non-existing types in the language, so they will not have corresponding rows
    """

    layers, sources, targets, runs, relations, accs = [], [], [], [], [], []
    header, types = None, None
    with open(filename) as f:
        for line in f:
            splt = line.strip('\n').split('\t')
            if header == None:
                header = splt
                types = splt[4:] 
                continue
            layer, source, target, run = splt[:4]
            for relation, acc in zip(types, splt[4:]):
                if acc != '':
                    layers.append(layer)
                    sources.append(source)
                    targets.append(target)
                    runs.append(run)
                    relations.append(relation)
                    accs.append(float(acc)*scale)
    df = pd.DataFrame({'layer': layers, 'source': sources, 'target': targets, 'run': runs, 'relation': relations, 'accuracy': accs})
    return df


def plot_pair_by_layer(ax, layers, all_accs, maj, mfl, title, hide_xlabel=False, hide_ylabel=False, 
                       ymin=0, ymax=100, plot_maj=True, nbins=6, delta_above=True, delta_val=4):
    
    # compute stats
    means = np.mean(all_accs, axis=0)
    stds = np.std(all_accs, axis=0)
    maxs = np.max(all_accs, axis=0)
    mins = np.max(all_accs, axis=0)
    deltas = [0] + [means[i+1]-means[i] for i in range(len(means)-1)]
        
    num_runs = len(all_accs)
    flat_accs = np.concatenate(all_accs)
    df = pd.DataFrame({'Layer' : [0,1,2,3,4]*num_runs, 'Accuracy' : flat_accs })
    ax.set_ylim(ymin,ymax)
    sns.swarmplot(x='Layer', y='Accuracy', data=df, ax=ax)
    if hide_xlabel:
        ax.set_xlabel('')
    if hide_ylabel:
        ax.set_ylabel('')
    if plot_maj:
        maj_line = ax.axhline(y=maj, label='Majority', linestyle='--', color='black')
    else:
        maj_line = None
    mfl_line = ax.axhline(y=mfl, label='MFL', linestyle='-.', color='black')

    for i in range(len(deltas)):
        if delta_above:
            x, y = i, maxs[i] + delta_val
        else:
            x, y = i, mins[i] - delta_val*2
        str_val = '{:+.1f} ({:.1f})'.format(deltas[i], stds[i])
        ax.text(x, y, str_val, horizontalalignment='center', size='small')
    xmin, xmax = plt.xlim()
    #ax.text(xmax-0.4, maj+1, 'maj', horizontalalignment='left', size='medium')
    #ax.text(xmax-0.4, mfl+1, 'mfl', horizontalalignment='left', size='medium')
    
    ax.locator_params(axis='y', nbins=nbins) 
    
    ax.set_title(title)
    #ax.tight_layout()
    #plt.savefig(figname)
    
    return maj_line, mfl_line


def plot_pairs_by_layer(df, pairs, majs, mfls, figname, fignum, ymin=0, plot_maj=True, 
                        legend=True, xsubs=3, ysubs=4, scale_figx=1.6, scale_figy=1.6, 
                        delta_val=4, suptitle=None):
    
    fig = plt.figure(fignum)
    #default_size = fig.get_size_inches() 
    #fig.set_size_inches( (default_size[0]*2, default_size[1]*2) )
    
    xsubs, ysubs = xsubs, ysubs
    f, axarr = plt.subplots(ysubs, xsubs, sharex=True, sharey=True)

    default_size = f.get_size_inches() 
    f.set_size_inches( (default_size[0]*scale_figx, default_size[1]*scale_figy) )
        
    maj_line, mfl_line = '', ''
    for i, ((source, target), maj, mfl)  in enumerate(zip(pairs, majs, mfls)):
        ax = f.axes[i]
        df_source_target = df[(df['source'] == source) & (df['target'] == target)]
        accs = get_accs_from_df(df_source_target)
        layers = df_source_target.layer.values
        hide_xlabel = True if i < (ysubs-1)*xsubs else False
        hide_ylabel = True if i % xsubs > 0 else False
        maj_line, mfl_line = plot_pair_by_layer(ax, layers, accs, maj, mfl, pretty_lang_names[source] + u"\u2192" + pretty_lang_names[target], 
                                                hide_xlabel=hide_xlabel, hide_ylabel=hide_ylabel, ymin=ymin, plot_maj=plot_maj, delta_val=delta_val)
    
    if legend:
        # hide unused axes
        axarr[-1, -1].axis('off')
        #for ax in f.axes[len(pairs):-1]:
        #    ax.axis('off')
                    
        if plot_maj:
            f.legend([maj_line, mfl_line], ['maj', 'mfl'], loc='lower left', bbox_to_anchor=(0.8,0.1), markerscale=1.5, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)
        else:
            f.legend([mfl_line], ['mfl'], loc='lower left', bbox_to_anchor=(0.8,0.1), markerscale=1.5, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)        

    if suptitle:
        print('suptitle: ', suptitle)
        plt.suptitle(suptitle)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()
    plt.savefig(figname)
    return fignum + 1


def plot_pairs_by_layer_semdeprel_schemes(df, pairs, majs, mfls, figname, fignum, ymin=0, ymax=100, plot_maj=True):
    
    fig = plt.figure(fignum)
    #default_size = fig.get_size_inches() 
    #fig.set_size_inches( (default_size[0]*2, default_size[1]*2) )
    
    xsubs, ysubs = 3, 6
    f, axarr = plt.subplots(ysubs, xsubs, sharex=True, sharey=True)

    default_size = f.get_size_inches() 
    f.set_size_inches( (default_size[0]*1.8, default_size[1]*1.8) )
    
    schemes = df.scheme.unique()
        
    maj_line, mfl_line = '', ''
    subplot_counter = 0 
    for s in range(len(schemes)):
        for (source, target), maj, mfl  in zip(pairs[s], majs[s], mfls[s]):            
            ax = f.axes[subplot_counter]
            df_source_target_scheme = df[(df['source'] == source) & (df['target'] == target) & (df['scheme'] == schemes[s])]
            accs = get_accs_from_df(df_source_target_scheme)
            layers = df_source_target_scheme.layer.values
            hide_xlabel = True if subplot_counter < (ysubs-1)*xsubs else False
            hide_ylabel = True if subplot_counter % xsubs > 0 else False
            maj_line, mfl_line = plot_pair_by_layer(ax, layers, accs, maj, mfl, pretty_lang_names[source] + u"\u2192" + pretty_lang_names[target] + ' ({})'.format(schemes[s]), 
                                                    hide_xlabel=hide_xlabel, hide_ylabel=hide_ylabel, ymin=ymin, ymax=ymax, plot_maj=plot_maj, nbins=3, delta_above=False)
            subplot_counter += 1
    
    # hide unused axes
    #axarr[-1, -1].axis('off')
    #for ax in f.axes[len(pairs):-1]:
    #    ax.axis('off')
                
#    if plot_maj:
#        f.legend([maj_line, mfl_line], ['maj', 'mfl'], loc='lower left', bbox_to_anchor=(0.8,0.1), markerscale=1.5, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)
#    else:
#        f.legend([mfl_line], ['mfl'], loc='lower left', bbox_to_anchor=(0.8,0.1), markerscale=1.5, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)        
    plt.tight_layout()
    plt.savefig(figname)
    return fignum + 1


def plot_pairs_by_layer_semdeprel_schemes2(df, pairs, majs, mfls, figname, fignum, ymin=0, ymax=100, plot_maj=True):
    
    fig = plt.figure(fignum)
    default_size = fig.get_size_inches() 
    fig.set_size_inches( (default_size[0]*2.5, default_size[1]*2.5) )
    
    outer = gridspec.GridSpec(3, 1, wspace=0.2, hspace=0.5)
    
    
    xsubs, ysubs = 3, 2
    #f, _ = plt.subplots(ysubs, xsubs) #, sharex=True, sharey=True)

    #default_size = f.get_size_inches() 
    #f.set_size_inches( (default_size[0]*1.8, default_size[1]*1.8) )
    
    schemes = df.scheme.unique()
        
    maj_line, mfl_line = '', ''
    for s in range(3):
        inner = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[s], wspace=0.5, hspace=1)
        for i, ((source, target), maj, mfl)  in enumerate(zip(pairs[s], majs[s], mfls[s])): 
            ax = plt.Subplot(fig, inner[i])
            df_source_target_scheme = df[(df['source'] == source) & (df['target'] == target) & (df['scheme'] == schemes[s])]
            accs = get_accs_from_df(df_source_target_scheme)
            layers = df_source_target_scheme.layer.values
            hide_xlabel = True if i < (ysubs-1)*xsubs else False
            hide_ylabel = True if i % xsubs > 0 else False
            maj_line, mfl_line = plot_pair_by_layer(ax, layers, accs, maj, mfl, pretty_lang_names[source] + u"\u2192" + pretty_lang_names[target], 
                                                    hide_xlabel=hide_xlabel, hide_ylabel=hide_ylabel, ymin=ymin, ymax=ymax, plot_maj=plot_maj, nbins=3, delta_above=False)
            fig.add_subplot(ax)
    
    # hide unused axes
    #axarr[-1, -1].axis('off')
    #for ax in f.axes[len(pairs):-1]:
    #    ax.axis('off')
                
#    if plot_maj:
#        f.legend([maj_line, mfl_line], ['maj', 'mfl'], loc='lower left', bbox_to_anchor=(0.8,0.1), markerscale=1.5, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)
#    else:
#        f.legend([mfl_line], ['mfl'], loc='lower left', bbox_to_anchor=(0.8,0.1), markerscale=1.5, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)        
    #plt.tight_layout()
    plt.savefig(figname)
    return fignum + 1


def plot_pair_by_relation_type(labels, diffs_2_1, diffs_3_1, diffs_4_1, proportions, fignum, filename, title, legendloc='center right', xlim=None):
    
    xs = np.arange(1, len(labels)+1)
    
    fig = plt.figure(fignum) 
    default_size = fig.get_size_inches() 
    fig.set_size_inches( (default_size[0]*1.3, default_size[1]*1.3) )
    
    plt.scatter(diffs_2_1, xs, c=proportions, s=100, cmap='Greens', edgecolors='black', linewidths=1, label='Layer 2', alpha=0.7, hatch='////')
    plt.scatter(diffs_3_1, xs, c=proportions, s=100, cmap='Reds', edgecolors='black', linewidths=1, label='Layer 3', alpha=0.7, hatch='---')
    plt.scatter(diffs_4_1, xs, c=proportions, s=100, cmap='Blues', edgecolors='black', linewidths=1, label='Layer 4', alpha=0.7, hatch='\\\\\\\\')
    plt.yticks(xs, labels)
    plt.xlabel('Change in F1 score w.r.t layer 1', fontsize='large')
    if xlim:
        plt.xlim(xlim)
    plt.gca().invert_yaxis()
        
    plt.legend(loc=legendloc, markerscale=1.5, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)
    ax = plt.gca()
    legend = ax.get_legend()
    legend.legendHandles[0].set_color(plt.cm.Greens(.8))
    legend.legendHandles[1].set_color(plt.cm.Reds(.8))
    legend.legendHandles[2].set_color(plt.cm.Blues(.8))
    #legend.get_frame().set_alpha(0.5)
    
    plt.title(title, fontsize='large')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=(200))
    
    return fignum + 1


def plot_pairs_by_relation_type(results_types_file_pref, results_types_file_suf, pairs, source2proportions, figname_pref, fignum, xlim=None):
    
    cur_fignum = fignum
    for source, target in pairs:
        #print(source, target)
        df = load_data_by_type(results_types_file_pref + source + '-' + target + results_types_file_suf)
        layers, runs, types = df.layer.unique(), df.run.unique(), df.columns.values[4:]
        num_layers, num_runs, num_types = len(layers), len(runs), len(types)

        df_source_target = df[(df['source'] == source) & (df['target'] == target)]        
        f1 = np.empty((num_layers, num_runs, num_types))
        for l in range(len(layers)):
            #print(l)
            for r in range(len(runs)):
                #print(r)
                df_source_target_layer_run = df_source_target[(df_source_target['run'] == runs[r]) & (df_source_target['layer'] == layers[l])]
                f1[l][r] = df_source_target_layer_run.values[0][4:] # get all F1 scores
        
        diffs = np.empty((num_layers, num_runs, num_types)) # diff from layer 1
        for l in range(num_layers):
            for r in range(num_runs):
                diffs[l][r] = f1[l][r] - f1[1][r]
        diffs_mean = diffs.mean(axis=1) # dimensions: layer, type
        cur_fignum = plot_pair_by_relation_type(types, diffs_mean[2], diffs_mean[3], diffs_mean[4], 
                                                source2proportions[source], cur_fignum, 
                                                figname_pref + '.' + source + '-' + target + '.png', 
                                                pretty_lang_names[source] + u"\u2192" + pretty_lang_names[target], 
                                                legendloc='best', xlim=xlim)    
    return cur_fignum + 1


def plot_pair_by_relation_distance_wrt_layer(labels, diffs_2_1, diffs_3_1, diffs_4_1, diffs_2_1_std, diffs_3_1_std, diffs_4_1_std, 
                                             proportions, fignum, filename, title, plot_lines=False):
    
    xs = np.arange(1, len(labels)+1) 
    
    #matplotlib.rcParams.update({'font.family': 'sans-serif'})
    fig = plt.figure(fignum) 
    #default_size = fig.get_size_inches() 
    #fig.set_size_inches( (default_size[0]*1.3, default_size[1]*1.3) )
    
    if plot_lines:
        plt.errorbar(xs, diffs_2_1, yerr=diffs_2_1_std, label='Layer 2', alpha=0.7, marker='', color=plt.cm.Greens(.8))
        plt.errorbar(xs, diffs_3_1, yerr=diffs_3_1_std, label='Layer 3', alpha=0.7, marker='', color=plt.cm.Reds(.8))
        plt.errorbar(xs, diffs_4_1, yerr=diffs_4_1_std, label='Layer 4', alpha=0.7, marker='', color=plt.cm.Blues(.8))            
    else:
        plt.scatter(xs, diffs_2_1, c=proportions, s=100, cmap='Greens', edgecolors='black', linewidths=1, label='Layer 2', alpha=0.7)
        plt.scatter(xs, diffs_3_1, c=proportions, s=100, cmap='Reds', edgecolors='black', linewidths=1, label='Layer 3', alpha=0.7)
        plt.scatter(xs, diffs_4_1, c=proportions, s=100, cmap='Blues', edgecolors='black', linewidths=1, label='Layer 4', alpha=0.7)

    plt.xticks(xs, labels)
    plt.ylabel('Increase in accuracy w.r.t layer 1', fontsize='large')
    plt.xlabel('Relation distance', fontsize='large')
    
    plt.legend(loc='upper left', markerscale=1, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)
    ax = plt.gca()
    legend = ax.get_legend()
    legend.legendHandles[0].set_color(plt.cm.Greens(.8))
    legend.legendHandles[1].set_color(plt.cm.Reds(.8))
    legend.legendHandles[2].set_color(plt.cm.Blues(.8))
    if plot_lines:
        pass
        #legend.legendHandles[0]._legmarker.set_color(plt.cm.Greens(.8))
        #legend.legendHandles[1]._legmarker.set_color(plt.cm.Reds(.8))
        #legend.legendHandles[2]._legmarker.set_color(plt.cm.Blues(.8))    
    #legend.get_frame().set_alpha(0.5)
    
    plt.title(title, fontsize='large')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=(200))



def plot_pairs_by_relation_distance_wrt_layer(df, pairs, figname, fignum, subplot_dims=(3,4)):
    
    #fig = plt.figure(fignum)
    plt.figure(fignum)
    #default_size = fig.get_size_inches() 
    #fig.set_size_inches( (default_size[0]*2, default_size[1]*2) )
    
    xsubs, ysubs = subplot_dims
    f, axarr = plt.subplots(ysubs, xsubs, sharex=True, sharey=True)

    default_size = f.get_size_inches() 
    f.set_size_inches( (default_size[0]*1.6, default_size[1]*1.6) )
        
    layers, runs, dists = df.layer.unique(), df.run.unique(), [d for d in df.columns if d.startswith('dist')]
    dist_names = [pretty_dist_names.get(d, d) for d in dists]
    num_dists = len(dists)
    num_layers = len(layers)
    num_runs = len(runs)
    line2, line3, line4 = '', '', ''
    for i, (source, target),  in enumerate(pairs):
        ax = f.axes[i]
        df_source_target = df[(df['source'] == source) & (df['target'] == target)]
        accs = np.empty((num_layers,num_runs,num_dists)) # dimensions: layer, run, dist
        for l in range(len(layers)):
            for r in range(len(runs)):
                accs[l][r] = get_accs_from_df(df_source_target[(df_source_target['run'] == runs[r]) & (df_source_target['layer'] == layers[l])], 'dist')
        diffs = np.empty((num_layers, num_runs, num_dists)) # diff from layer 1
        for l in range(len(layers)):
            for r in range(len(runs)):
                diffs[l][r] = accs[l][r] - accs[1][r]
        diffs_mean = diffs.mean(axis=1) # dimensions: layer, dist
        diffs_std = diffs.std(axis=1)
        
        hide_xlabel = True if i < (ysubs-1)*xsubs else False
        hide_ylabel = True if i % xsubs > 0 else False
        line2, line3, line4 = plot_pair_by_relation_distance_wrt_layer_subplot(ax, dist_names, diffs_mean[2], diffs_mean[3], diffs_mean[4], 
                                                        diffs_std[2], diffs_std[3], diffs_std[4], 
                                                        [], pretty_lang_names[source] + u"\u2192" + pretty_lang_names[target], 
                                                        hide_xlabel=hide_xlabel, hide_ylabel=hide_ylabel)
    
    # hide unused axes
    axarr[-1, -1].axis('off')
    #for ax in f.axes[len(pairs):-1]:
    #    ax.axis('off')
                
    legend = f.legend([line2, line3, line4], ['Layer 2', 'Layer 3', 'Layer 4'], loc='lower left', bbox_to_anchor=(0.8,0.1), markerscale=1.5, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)    
    legend.legendHandles[0].set_color(plt.cm.Greens(.8))
    legend.legendHandles[1].set_color(plt.cm.Reds(.8))
    legend.legendHandles[2].set_color(plt.cm.Blues(.8))
    plt.tight_layout()
    plt.savefig(figname)
    return fignum + 1


def plot_pair_by_relation_distance_wrt_layer_subplot(ax, labels, diffs_2_1, diffs_3_1, diffs_4_1, diffs_2_1_std, diffs_3_1_std, diffs_4_1_std, 
                                             proportions, title, hide_xlabel=False, hide_ylabel=False): #, plot_lines=False):
    
    xs = np.arange(1, len(labels)+1) 
    
    #matplotlib.rcParams.update({'font.family': 'sans-serif'})
    #fig = plt.figure(fignum) 
    #default_size = fig.get_size_inches() 
    #fig.set_size_inches( (default_size[0]*1.3, default_size[1]*1.3) )
    
    #if plot_lines:
    if True:
        line2 = ax.errorbar(xs, diffs_2_1, yerr=diffs_2_1_std, label='Layer 2', alpha=0.7, marker='', color=plt.cm.Greens(.8))
        line3 = ax.errorbar(xs, diffs_3_1, yerr=diffs_3_1_std, label='Layer 3', alpha=0.7, marker='', color=plt.cm.Reds(.8))
        line4 = ax.errorbar(xs, diffs_4_1, yerr=diffs_4_1_std, label='Layer 4', alpha=0.7, marker='', color=plt.cm.Blues(.8))            
    else:
        line2 = ax.scatter(xs, diffs_2_1, c=proportions, s=100, cmap='Greens', edgecolors='black', linewidths=1, label='Layer 2', alpha=0.7)
        line3 = ax.scatter(xs, diffs_3_1, c=proportions, s=100, cmap='Reds', edgecolors='black', linewidths=1, label='Layer 3', alpha=0.7)
        line4 = ax.scatter(xs, diffs_4_1, c=proportions, s=100, cmap='Blues', edgecolors='black', linewidths=1, label='Layer 4', alpha=0.7)

    if hide_xlabel:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Relation distance', fontsize='large')
    if hide_ylabel:
        ax.set_ylabel('')   
    else:
        ax.set_ylabel('Accuracy ' + u"\u0394", fontsize='large')
    #ax.set_ylim(ymin=0)
    ax.locator_params(axis='y', nbins=6) 
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    
    #ax.legend(loc='upper left', markerscale=1, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)
    #ax = plt.gca()
    #legend = ax.get_legend()
    #legend.legendHandles[0].set_color(plt.cm.Greens(.8))
    #legend.legendHandles[1].set_color(plt.cm.Reds(.8))
    #legend.legendHandles[2].set_color(plt.cm.Blues(.8))
    #if plot_lines:
    if True:
        pass
        #legend.legendHandles[0]._legmarker.set_color(plt.cm.Greens(.8))
        #legend.legendHandles[1]._legmarker.set_color(plt.cm.Reds(.8))
        #legend.legendHandles[2]._legmarker.set_color(plt.cm.Blues(.8))    
    #legend.get_frame().set_alpha(0.5)
    
    ax.set_title(title, fontsize='large')
    
    #plt.tight_layout()
    #plt.savefig(filename, dpi=(200))    
    
    return line2, line3, line4
    
    
def plot_pair_by_relation_distance_wrt_distance(labels, diffs_layer_0, diffs_layer_1, diffs_layer_2, diffs_layer_3, diffs_layer_4, 
                                                diffs_layer_0_std, diffs_layer_1_std, diffs_layer_2_std, diffs_layer_3_std, diffs_layer_4_std,
                                                proportions, fignum, filename, title, plot_lines=False):
    
    xs = np.arange(1, len(labels)+1) 
    
    #matplotlib.rcParams.update({'font.family': 'sans-serif'})
    fig = plt.figure(fignum) 
    #default_size = fig.get_size_inches() 
    #fig.set_size_inches( (default_size[0]*1.3, default_size[1]*1.3) )
    
    if plot_lines:
        plt.errorbar(xs, diffs_layer_0, yerr=diffs_layer_0_std, label='Layer 0', alpha=0.7, marker='', color=plt.cm.Oranges(.8))
        plt.errorbar(xs, diffs_layer_1, yerr=diffs_layer_1_std, label='Layer 1', alpha=0.7, marker='', color=plt.cm.Purples(.8))
        plt.errorbar(xs, diffs_layer_2, yerr=diffs_layer_2_std, label='Layer 2', alpha=0.7, marker='', color=plt.cm.Greens(.8))
        plt.errorbar(xs, diffs_layer_3, yerr=diffs_layer_3_std, label='Layer 3', alpha=0.7, marker='', color=plt.cm.Reds(.8))
        plt.errorbar(xs, diffs_layer_4, yerr=diffs_layer_4_std, label='Layer 4', alpha=0.7, marker='', color=plt.cm.Blues(.8))
    
    else:    
        plt.scatter(xs, diffs_layer_0, c=proportions, s=100, cmap='Oranges', edgecolors='black', linewidths=1, label='Layer 0', alpha=0.7)
        plt.scatter(xs, diffs_layer_1, c=proportions, s=100, cmap='Purples', edgecolors='black', linewidths=1, label='Layer 1', alpha=0.7)            
        plt.scatter(xs, diffs_layer_2, c=proportions, s=100, cmap='Greens', edgecolors='black', linewidths=1, label='Layer 2', alpha=0.7)
        plt.scatter(xs, diffs_layer_3, c=proportions, s=100, cmap='Reds', edgecolors='black', linewidths=1, label='Layer 3', alpha=0.7)
        plt.scatter(xs, diffs_layer_4, c=proportions, s=100, cmap='Blues', edgecolors='black', linewidths=1, label='Layer 4', alpha=0.7)

    plt.xticks(xs, labels)
    plt.ylabel('Decrease in accuracy w.r.t distance 1', fontsize='large')
    plt.xlabel('Relation distance', fontsize='large')
    
    plt.legend(loc='upper left', markerscale=1, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)
    ax = plt.gca()
    legend = ax.get_legend()
    legend.legendHandles[0].set_color(plt.cm.Oranges(.8))
    legend.legendHandles[1].set_color(plt.cm.Purples(.8))
    legend.legendHandles[2].set_color(plt.cm.Greens(.8))
    legend.legendHandles[3].set_color(plt.cm.Reds(.8))
    legend.legendHandles[4].set_color(plt.cm.Blues(.8))
    if plot_lines:
        pass
        #legend.legendHandles[0]._legmarker.set_color(plt.cm.Oranges(.8))
        #legend.legendHandles[1]._legmarker.set_color(plt.cm.Purples(.8))
        #legend.legendHandles[2]._legmarker.set_color(plt.cm.Greens(.8))
        #legend.legendHandles[3]._legmarker.set_color(plt.cm.Reds(.8))
        #legend.legendHandles[4]._legmarker.set_color(plt.cm.Blues(.8))    
        
    #legend.get_frame().set_alpha(0.5)
    
    plt.title(title, fontsize='large')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=(200))


def plot_pair_by_relation_distance_wrt_distance_subplot(ax, labels, diffs_layer_0, diffs_layer_1, diffs_layer_2, diffs_layer_3, diffs_layer_4, 
                                                        diffs_layer_0_std, diffs_layer_1_std, diffs_layer_2_std, diffs_layer_3_std, diffs_layer_4_std, 
                                                        proportions, title, hide_xlabel=False, hide_ylabel=False): #, plot_lines=False):
    
    xs = np.arange(1, len(labels)+1) 
    
    #matplotlib.rcParams.update({'font.family': 'sans-serif'})
    #fig = plt.figure(fignum) 
    #default_size = fig.get_size_inches() 
    #fig.set_size_inches( (default_size[0]*1.3, default_size[1]*1.3) )
    
    #if plot_lines:
    if True:
        line0 = ax.errorbar(xs, diffs_layer_0, yerr=diffs_layer_0_std, label='Layer 0', alpha=0.7, marker='', color=plt.cm.Oranges(.8))
        line1 = ax.errorbar(xs, diffs_layer_1, yerr=diffs_layer_1_std, label='Layer 1', alpha=0.7, marker='', color=plt.cm.Purples(.8))
        line2 = ax.errorbar(xs, diffs_layer_2, yerr=diffs_layer_2_std, label='Layer 2', alpha=0.7, marker='', color=plt.cm.Greens(.8))
        line3 = ax.errorbar(xs, diffs_layer_3, yerr=diffs_layer_3_std, label='Layer 3', alpha=0.7, marker='', color=plt.cm.Reds(.8))
        line4 = ax.errorbar(xs, diffs_layer_4, yerr=diffs_layer_4_std, label='Layer 4', alpha=0.7, marker='', color=plt.cm.Blues(.8))         
    else:
        line0 = ax.scatter(xs, diffs_layer_0, c=proportions, s=100, cmap='Oranges', edgecolors='black', linewidths=1, label='Layer 0', alpha=0.7)
        line1 = ax.scatter(xs, diffs_layer_1, c=proportions, s=100, cmap='Purples', edgecolors='black', linewidths=1, label='Layer 1', alpha=0.7)            
        line2 = ax.scatter(xs, diffs_layer_2, c=proportions, s=100, cmap='Greens', edgecolors='black', linewidths=1, label='Layer 2', alpha=0.7)
        line3 = ax.scatter(xs, diffs_layer_3, c=proportions, s=100, cmap='Reds', edgecolors='black', linewidths=1, label='Layer 3', alpha=0.7)
        line4 = ax.scatter(xs, diffs_layer_4, c=proportions, s=100, cmap='Blues', edgecolors='black', linewidths=1, label='Layer 4', alpha=0.7)

    if hide_xlabel:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Relation distance', fontsize='large')
    if hide_ylabel:
        ax.set_ylabel('')   
    else:
        ax.set_ylabel('Accuracy ' + u"\u0394", fontsize='large')
    #ax.set_ylim(ymin=0)
    ax.locator_params(axis='y', nbins=6) 
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    
    #ax.legend(loc='upper left', markerscale=1, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)
    #ax = plt.gca()
    #legend = ax.get_legend()
    #legend.legendHandles[0].set_color(plt.cm.Greens(.8))
    #legend.legendHandles[1].set_color(plt.cm.Reds(.8))
    #legend.legendHandles[2].set_color(plt.cm.Blues(.8))
    #if plot_lines:
    if True:
        pass
        #legend.legendHandles[0]._legmarker.set_color(plt.cm.Greens(.8))
        #legend.legendHandles[1]._legmarker.set_color(plt.cm.Reds(.8))
        #legend.legendHandles[2]._legmarker.set_color(plt.cm.Blues(.8))    
    #legend.get_frame().set_alpha(0.5)
    
    ax.set_title(title, fontsize='large')
    
    #plt.tight_layout()
    #plt.savefig(filename, dpi=(200))    
    
    return line0, line1, line2, line3, line4
    #return line1, line2, line3, line4
    

def plot_pairs_by_relation_distance_wrt_distance(df, pairs, figname, fignum, subplot_dims=(3,4)):
    
    #fig = plt.figure(fignum)
    plt.figure(fignum)
    #default_size = fig.get_size_inches() 
    #fig.set_size_inches( (default_size[0]*2, default_size[1]*2) )
    
    xsubs, ysubs = subplot_dims
    f, axarr = plt.subplots(ysubs, xsubs, sharex=True, sharey=True)

    default_size = f.get_size_inches() 
    f.set_size_inches( (default_size[0]*1.6, default_size[1]*1.6) )
        
    layers, runs, dists = df.layer.unique(), df.run.unique(), [d for d in df.columns if d.startswith('dist')]
    dist_names = [pretty_dist_names.get(d, d) for d in dists]
    num_dists = len(dists)
    num_layers = len(layers)
    num_runs = len(runs)
    line0, line1, line2, line3, line4 = '', '', '', '', ''
    for i, (source, target),  in enumerate(pairs):
        ax = f.axes[i]
        df_source_target = df[(df['source'] == source) & (df['target'] == target)]
        accs = np.empty((num_layers,num_runs,num_dists)) # dimensions: layer, run, dist
        for l in range(len(layers)):
            for r in range(len(runs)):
                accs[l][r] = get_accs_from_df(df_source_target[(df_source_target['run'] == runs[r]) & (df_source_target['layer'] == layers[l])], 'dist')
        diffs = np.empty((num_layers, num_runs, num_dists)) # diff from distance 1
        for l in range(len(layers)):
            for r in range(len(runs)):
                diffs[l][r] = accs[l][r] - accs[l][r][0]
        diffs_mean = diffs.mean(axis=1) # dimensions: layer, dist
        diffs_std = diffs.std(axis=1)
        
        hide_xlabel = True if i < (ysubs-1)*xsubs else False
        hide_ylabel = True if i % xsubs > 0 else False
        line0, line1, line2, line3, line4 = plot_pair_by_relation_distance_wrt_distance_subplot(ax, dist_names, 
        #line1, line2, line3, line4 = plot_pair_by_relation_distance_wrt_distance_subplot(ax, dist_names, 
                                                        diffs_mean[0], diffs_mean[1], diffs_mean[2], diffs_mean[3], diffs_mean[4], 
                                                        diffs_std[0], diffs_std[1], diffs_std[2], diffs_std[3], diffs_std[4], 
                                                        [], pretty_lang_names[source] + u"\u2192" + pretty_lang_names[target], 
                                                        hide_xlabel=hide_xlabel, hide_ylabel=hide_ylabel)
    
    # hide unused axes
    axarr[-1, -1].axis('off')
    #for ax in f.axes[len(pairs):-1]:
    #    ax.axis('off')
                
    legend = f.legend([line0, line1, line2, line3, line4], ['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'], loc='lower left', bbox_to_anchor=(0.8,0.05), markerscale=1.5, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)    
    #legend = f.legend([line1, line2, line3, line4], ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'], loc='lower left', bbox_to_anchor=(0.8,0.05), markerscale=1.5, fontsize='medium', frameon=True, title='Legend', edgecolor='black', labelspacing=1)    
    legend.legendHandles[0].set_color(plt.cm.Oranges(.8))
    legend.legendHandles[1].set_color(plt.cm.Purples(.8))
    legend.legendHandles[2].set_color(plt.cm.Greens(.8))
    legend.legendHandles[3].set_color(plt.cm.Reds(.8))
    legend.legendHandles[4].set_color(plt.cm.Blues(.8))
    plt.tight_layout()
    plt.savefig(figname)
    return fignum + 1


def plot_averages(df, figname, fignum, use_en_source=True, num_accs=3):

    plt.figure(fignum)
    if use_en_source:
        df_side = df[(df.source == 'en') & (df.target != 'en')]
        layers = np.concatenate([[i]*5 for i in range(5)] * num_accs)
    else:
        df_side = df[(df.source != 'en') & (df.target == 'en')]
        layers = list(range(5))*5*num_accs

    accs = get_accs_from_df(df_side, col_pref='acc')
    flat_accs = np.concatenate(accs)
    df_plot = pd.DataFrame({'Layer' : layers, 'Accuracy' : flat_accs }) 
    #print(df_plot)
    sns.boxplot(x='Layer', y='Accuracy', data=df_plot)

    plt.tight_layout()
    plt.savefig(figname)
    return fignum + 1
    

def plot_averages_by_distance(df, figname, fignum, use_en_source=True, num_accs=24, pointplot=True, hue='Distance'):

    plt.figure(fignum)
    if use_en_source:
        df_side = df[(df.source == 'en') & (df.target != 'en')]
        layers = np.concatenate([[i]*5 for i in range(5)] * num_accs)        
    else:
        df_side = df[(df.source != 'en') & (df.target == 'en')]
        layers = list(range(5))*5*num_accs

    accs = get_accs_from_df(df_side, col_pref='dist')
    flat_accs = np.concatenate(accs)
    dists = np.concatenate([[pretty_dist_names_list[i]]*75 for i in range(8)])
    df_plot = pd.DataFrame({'Layer' : layers, 'Accuracy' : flat_accs, 'Distance' : dists }) 
    #print(df_plot)
    plotfunc = sns.pointplot if pointplot else sns.boxplot
    if hue == 'Distance':
        plotfunc(x='Layer', y='Accuracy', data=df_plot, hue='Distance')
    else:
        plotfunc(x='Distance', y='Accuracy', data=df_plot, hue='Layer')
        plt.xticks(range(8), pretty_dist_names_list)
    

    plt.tight_layout()
    plt.savefig(figname)
    return fignum + 1


def plot_averages_by_type(df, figname, fignum, use_en_source=True, pointplot=True, layer0=True):

    plt.figure(fignum)
    if use_en_source:
        df_side = df[(df.source == 'en') & (df.target != 'en')]
    else:
        df_side = df[(df.source != 'en') & (df.target == 'en')]
    if not layer0:
        df_side = df_side[df_side.layer != '0']


    plotfunc = sns.pointplot if pointplot else sns.boxplot
    if pointplot:
        plotfunc(x='accuracy', y='relation', hue='layer', data=df_side, join=False)
    else:
        plotfunc(x='accuracy', y='relation', hue='layer', data=df_side)
    plt.xlabel('Accuracy')
    plt.ylabel('')

    plt.tight_layout()
    plt.savefig(figname)
    return fignum + 1
    

data_file = os.path.join(RESULTS_DIR, 'results-un-deprel-plot.txt')
df = load_data(data_file)
fignum = 0
#fignum = plot_pairs_by_layer(df, pairs, majs, mfls, 'all-deprel.png', fignum, ymin=30, plot_maj=False)

fignum = plot_averages(df, 'all-deprel-en-to-ave.png', fignum, use_en_source=True)
fignum = plot_averages(df, 'all-deprel-to-en-ave.png', fignum, use_en_source=False)


data_file_dists = os.path.join(RESULTS_DIR, 'results-un-deprel-distance-nopunct-plot.txt')
df = load_data_by_distance(data_file_dists)
#plot_pairs_by_relation_distance_wrt_layer(df, pairs, 'dists-pairs-layer-nopunct.png', 10)
#plot_pairs_by_relation_distance_wrt_distance(df, pairs, 'dists-pairs-distance-nopunct.png', 11)



#df_en_ar = df[(df['source'] == 'en') & (df['target'] == 'ar')]
#accs = get_accs_from_df(df_en_ar)
#layers = df_en_ar.layer.values
#plot_all_by_layer(layers, accs, en_maj, en_mfl, 'tmp2.png', 0, 'en->ar')
    


labels = ['acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'aux:pass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'cop', 'det', 'discourse', 'flat', 'list', 'mark', 'nmod', 'nmod:poss', 'nsubj', 'nsubj:pass', 'nummod', 'obj', 'obl', 'parataxis', 'punct', 'xcomp']
f1_diffs_4_1 = [5.662, 9.082, 12.644, 0.468, 1.372, 10.498, 0.76, 2.872, 1.206, 0.268, 9.748, 5.416, 12.76, 1.104, -0.022, 4.192, 7.934, 12.16, 2.182, 8.08, 2.17, 3.808, 8.54, 5.366, 5.974, 7.738, 20.956, 1.1, 4.922]
f1_diffs_3_1 = [4.302, 5.154, 8.112, 1.612, 0.886, 7.06, 0.766, 2.698, 1.1, 0.25, 7.142, 3.758, 9.292, 0.95, -0.032, 4.644, 7.808, 7.516, 2.056, 5.408, 1.184, 3.23, 7.628, 3.886, 3.924, 5.622, 13.826, 0.762, 3.476]
f1_diffs_2_1 = [4.554, 3.832, 5.666, 0.26, 0.58, 3.084, 0.648, 1.744, 0.088, 0.036, 6.08, 2.556, 6.586, 0.558, -0.236, 2.922, 2.262, 1.452, 0.946, 3.05, 0.754, 2.22, 4.494, 1.946, 2.318, 4.136, 11.75, 0.058, 0.91]
proportions = [0.007735396106, 0.009558104383, 0.0166711123, 0.0570374322, 0.05210278296, 0.008268871699, 0.03667644705, 0.005201387037, 0.08793456033, 0.03338668089, 0.01080288077, 0.05130256957, 0.03841024273, 0.0248510714, 0.08139948431, 0.005423668534, 0.01098070597, 0.01115853116, 0.03476482618, 0.03432026318, 0.01711567529, 0.08780119143, 0.004534542545, 0.01240330755, 0.05245843336, 0.04676802703, 0.00902462879, 0.1363919267, 0.01551524851]


#plot_pair_by_relation_type(labels, f1_diffs_2_1, f1_diffs_3_1, f1_diffs_4_1, proportions, 2, 'types.png', 'Title')


### no punc ###
counts_en = [174, 215, 375, 1283, 1172, 186, 825, 117, 1978, 751, 9, 243, 1154, 89, 864, 559, 23, 1, 1, 1831, 22, 122, 66, 62, 247, 7, 16, 41, 251, 782, 772, 13, 385, 37, 1975, 102, 279, 1180, 1052, 48, 66, 1, 203, 3, 20, 349]
counts_ar = [559, 197, 322, 130, 2442, 81, 143, 13, 4356, 1955, 319, 1230, 97, 40, 101, 199, 84, 29, 187, 765, 6223, 1576, 74, 378, 1939, 1395, 6, 521, 169]
counts_es = [138, 188, 156, 391, 583, 186, 188, 28, 1736, 372, 82, 10, 483, 166, 34, 20, 1707, 126, 175, 233, 453, 965, 461, 21, 178, 440, 654, 68, 62]
counts_fr = [170, 75, 94, 458, 480, 118, 187, 50, 1282, 258, 60, 44, 331, 132, 1, 13, 1356, 5, 56, 178, 119, 27, 231, 770, 113, 573, 54, 146, 388, 563, 32, 45]
counts_ru = [112, 73, 61, 345, 1302, 321, 12, 69, 1219, 323, 26, 31, 6, 484, 66, 2, 174, 59, 33, 123, 1, 250, 107, 65, 52, 1333, 579, 84, 64, 85, 313, 923, 45, 72, 1, 78]
counts_zh = [484, 176, 15, 366, 188, 104, 90, 22, 41, 328, 117, 304, 78, 590, 190, 203, 229, 337, 190, 36, 1, 291, 426, 15, 11, 27, 7, 432, 12, 2, 236, 1326, 160, 970, 29, 621, 775, 235, 157]
proportions_en = [0.00872136735, 0.01077640219, 0.01879605032, 0.06430755351, 0.05874392261, 0.00932284096, 0.04135131071, 0.005864367701, 0.09914290011, 0.03764222345, 0.0004511052078, 0.01217984061, 0.05784171219, 0.004460929277, 0.04330609994, 0.02801864568, 0.00115282442, 0.00005012280086, 0.00005012280086, 0.09177484838, 0.001102701619, 0.006114981705, 0.003308104857, 0.003107613653, 0.01238033181, 0.000350859606, 0.0008019648138, 0.002055034835, 0.01258082302, 0.03919603027, 0.03869480227, 0.0006515964112, 0.01929727833, 0.001854543632, 0.0989925317, 0.005112525688, 0.01398426144, 0.05914490502, 0.05272918651, 0.002405894441, 0.003308104857, 0.00005012280086, 0.01017492858, 0.0001503684026, 0.001002456017, 0.0174928575]
proportions_ar = [0.02189580885, 0.007716412064, 0.01261261261, 0.00509204857, 0.09565217391, 0.003172737955, 0.005601253427, 0.000509204857, 0.1706227967, 0.07657657658, 0.0124951038, 0.0481786134, 0.003799451626, 0.001566784175, 0.003956130043, 0.007794751273, 0.003290246769, 0.001135918527, 0.00732471602, 0.02996474736, 0.2437524481, 0.06173129651, 0.002898550725, 0.01480611046, 0.07594986291, 0.05464159812, 0.0002350176263, 0.02040736389, 0.006619663141]
proportions_es = [0.01339285714, 0.007363885625, 0.006110458284, 0.01531531532, 0.02283587936, 0.007285546416, 0.007363885625, 0.001096748923, 0.06799843322, 0.01457109283, 0.00321190756, 0.0003916960439, 0.01891891892, 0.006502154328, 0.001331766549, 0.0007833920877, 0.06686251469, 0.004935370153, 0.006854680768, 0.009126517822, 0.01774383079, 0.03779866823, 0.01805718762, 0.0008225616921, 0.006972189581, 0.01723462593, 0.02561692127, 0.002663533098, 0.002428515472]
proportions_fr = [0.02021643477, 0.008919015341, 0.01117849923, 0.05446545368, 0.05708169818, 0.01403258414, 0.02223807825, 0.005946010227, 0.1524557022, 0.03068141277, 0.007135212273, 0.005232489, 0.0393625877, 0.015697467, 0.0001189202045, 0.001545962659, 0.1612557974, 0.0005946010227, 0.006659531454, 0.02116779641, 0.01415150434, 0.003210845523, 0.02747056725, 0.0915685575, 0.01343798311, 0.0681412772, 0.006421691045, 0.01736234986, 0.04614103936, 0.06695207516, 0.003805446545, 0.005351409204]
proportions_ru = [0.01259417519, 0.008208703475, 0.006859327561, 0.03879455752, 0.1464072866, 0.03609580569, 0.001349375914, 0.007758911503, 0.1370741032, 0.03632070168, 0.002923647813, 0.003485887777, 0.0006746879568, 0.05442482852, 0.007421567525, 0.0002248959856, 0.01956595075, 0.006634431575, 0.003710783763, 0.01383110311, 0.0001124479928, 0.0281119982, 0.01203193523, 0.007309119532, 0.005847295626, 0.1498931744, 0.06510738783, 0.009445631395, 0.007196671539, 0.009558079388, 0.03519622175, 0.1037894974, 0.005060159676, 0.008096255482, 0.0001124479928, 0.008770943439]
proportions_zh = [0.04928215049, 0.017920782, 0.001527339375, 0.03726708075, 0.0191426535, 0.010589553, 0.009164036249, 0.00224009775, 0.004174727624, 0.033397821, 0.01191324712, 0.030954078, 0.007942164749, 0.06007534874, 0.01934629875, 0.02066999287, 0.02331738112, 0.03431422462, 0.01934629875, 0.0036656145, 0.000101822625, 0.02963038387, 0.04337643824, 0.001527339375, 0.001120048875, 0.002749210875, 0.0007127583749, 0.04398737399, 0.0012218715, 0.00020364525, 0.0240301395, 0.1350168007, 0.01629162, 0.09876794624, 0.002952856125, 0.06323185012, 0.07891253437, 0.02392831687, 0.01598615212]
source2proportions = {'en': proportions_en, 'ar':proportions_ar, 'es':proportions_es, 'fr':proportions_fr, 'ru':proportions_ru, 'zh':proportions_zh}
types_file_pref = os.path.join(RESULTS_DIR, 'results-un-deprel-types-nopunct-')
types_file_suf = '-plot.txt'

fignum = 22
#plot_pairs_by_relation_type(types_file_pref, types_file_suf, pairs, source2proportions, 'types', fignum)

counts_en_min100 = [c for c in counts_en if c >= 100]
counts_ar_min100 = [c for c in counts_ar if c >= 100]
counts_es_min100 = [c for c in counts_es if c >= 100]
counts_fr_min100 = [c for c in counts_fr if c >= 100]
counts_ru_min100 = [c for c in counts_ru if c >= 100]
counts_zh_min100 = [c for c in counts_zh if c >= 100]
proportions_en_min100 = [c/np.sum(counts_en_min100) for c in counts_en_min100]
proportions_ar_min100 = [c/np.sum(counts_ar_min100) for c in counts_ar_min100]
proportions_es_min100 = [c/np.sum(counts_es_min100) for c in counts_es_min100]
proportions_fr_min100 = [c/np.sum(counts_fr_min100) for c in counts_fr_min100]
proportions_ru_min100 = [c/np.sum(counts_ru_min100) for c in counts_ru_min100]
proportions_zh_min100 = [c/np.sum(counts_zh_min100) for c in counts_zh_min100]

source2proportions_min100 = {'en': proportions_en_min100, 'ar':proportions_ar_min100, 'es':proportions_es_min100, 'fr':proportions_fr_min100, 'ru':proportions_ru_min100, 'zh':proportions_zh_min100}
types_min100_file_pref = os.path.join(RESULTS_DIR, 'results-un-deprel-types-nopunct-min100-')
types_min100_file_suf = '-plot.txt'

fignum = 222
#fignum = plot_pairs_by_relation_type(types_min100_file_pref, types_min100_file_suf, pairs, source2proportions_min100, 'types-min100', fignum)

types_all_file = os.path.join(RESULTS_DIR, 'results-un-deprel-types-nopunct-min80-plot.txt')
df = load_data_by_type_all(types_all_file)
#fignum = plot_averages_by_type(df, 'all-deprel-types-en-to-ave.png', fignum, use_en_source=True, pointplot=True)
#fignum = plot_averages_by_type(df, 'all-deprel-types-to-en-ave.png', fignum, use_en_source=False, pointplot=True)
#fignum = plot_averages_by_type(df, 'all-deprel-types-en-to-ave-box.png', fignum, use_en_source=True, pointplot=False)
#fignum = plot_averages_by_type(df, 'all-deprel-types-to-en-ave-box.png', fignum, use_en_source=False, pointplot=False)
#fignum = plot_averages_by_type(df, 'all-deprel-types-en-to-no0-ave.png', fignum, use_en_source=True, pointplot=True, layer0=False)
#fignum = plot_averages_by_type(df, 'all-deprel-types-to-en-no0-ave.png', fignum, use_en_source=False, pointplot=True, layer0=False)
#fignum = plot_averages_by_type(df, 'all-deprel-types-en-to-ave-no0-box.png', fignum, use_en_source=True, pointplot=False, layer0=False)
#fignum = plot_averages_by_type(df, 'all-deprel-types-to-en-ave-no0-box.png', fignum, use_en_source=False, pointplot=False, layer0=False)

types_all_deltas_file = os.path.join(RESULTS_DIR, 'results-un-deprel-types-nopunct-min80-deltas-plot.txt')
df = load_data_by_type_all(types_all_deltas_file)
#fignum = plot_averages_by_type(df, 'all-deprel-types-en-to-deltas-ave.png', fignum, use_en_source=True, pointplot=True)
#fignum = plot_averages_by_type(df, 'all-deprel-types-to-en-deltas-ave.png', fignum, use_en_source=False, pointplot=True)
#fignum = plot_averages_by_type(df, 'all-deprel-types-en-to-deltas-ave-box.png', fignum, use_en_source=True, pointplot=False)
#fignum = plot_averages_by_type(df, 'all-deprel-types-to-en-deltas-ave-box.png', fignum, use_en_source=False, pointplot=False)






labels = ['1', '2', '3', '4', '5', '6-7', '8-10', '>10']
diffs_4_1 = [1.179447684, 2.458264539, 4.215920738, 6.909090909, 8.509249184, 8.645038168, 10.64471879, 7.330016584]
diffs_3_1 = [1.056221807, 1.841863878, 3.143149983, 4.545454545, 6.245919478, 5.591603053, 7.709190672, 5.737976783]
diffs_2_1 = [0.2640554516, 0.8548890112, 1.510078579, 3.054545455, 5.048966268, 3.645038168, 5.788751715, 4.112769486]
proportions = [0.3948477345, 0.236804379, 0.1271558278, 0.07167991659, 0.03992354142, 0.04552760763, 0.03166949042, 0.05239150267]
diffs_4_1_std = [0.2]*len(diffs_4_1)
diffs_3_1_std = [0.2]*len(diffs_3_1)
diffs_2_1_std = [0.2]*len(diffs_2_1)

#plot_pair_by_relation_distance_wrt_layer(labels, diffs_2_1, diffs_3_1, diffs_4_1, diffs_2_1_std, diffs_3_1_std, diffs_4_1_std, proportions, 3, 
#                                         'dists_wrt_layer.png', 'Title', plot_lines=True)






labels = ['2', '3', '4', '5', '6-7', '8-10', '>10']
diffs_layer_0 = [9.669623283, 12.35963474, 18.25599776, 22.07196373, 24.67473475, 23.80776232, 13.15833456]
diffs_layer_1 = [0.9845003059, 3.541995954, 9.438528758, 11.91341912, 12.3124589, 13.94682156, 8.356665347]
diffs_layer_2 = [0.3936667463, 2.295972827, 6.648038755, 7.128508303, 8.931476183, 8.422125299, 4.507951313]
diffs_layer_3 = [0.1988582343, 1.455067778, 5.949296019, 6.723721448, 7.777077652, 7.293852697, 3.674910371]
diffs_layer_4 = [-0.2943165487, 0.5055229005, 3.708885533, 4.58361762, 4.846868415, 4.481550453, 2.206096448]
proportions = [0.3948477345, 0.236804379, 0.1271558278, 0.07167991659, 0.03992354142, 0.04552760763, 0.03166949042, 0.05239150267][1:]
diffs_layer_0_std = [0.2]*len(diffs_layer_0)
diffs_layer_1_std = [0.2]*len(diffs_layer_1)
diffs_layer_2_std = [0.2]*len(diffs_layer_2)
diffs_layer_3_std = [0.2]*len(diffs_layer_3)
diffs_layer_4_std = [0.2]*len(diffs_layer_4)

#plot_pair_by_relation_distance_wrt_distance(labels,  diffs_layer_0, diffs_layer_1, diffs_layer_2, diffs_layer_3, diffs_layer_4, 
#                                            diffs_layer_0_std, diffs_layer_1_std, diffs_layer_2_std, diffs_layer_3_std, diffs_layer_4_std, proportions, 4, 
#                                            'dists_wrt_dist.png', 'Title', plot_lines=True)

fignum = fignum + 1
df = load_data_by_distance(os.path.join(RESULTS_DIR, 'results-un-deprel-distance-nopunct-plot.txt'))
# num_accs = 8 distances * 3 runs
#fignum = plot_averages_by_distance(df, 'all-deprel-dist-en-to-ave-huedist.png', fignum, use_en_source=True, num_accs=8*3, pointplot=True, hue='Distance')
#fignum = plot_averages_by_distance(df, 'all-deprel-dist-to-en-ave-huedist.png', fignum, use_en_source=False, num_accs=8*3, pointplot=True, hue='Distance')
#fignum = plot_averages_by_distance(df, 'all-deprel-dist-en-to-ave-huelayer.png', fignum, use_en_source=True, num_accs=8*3, pointplot=True, hue='Layer')
#fignum = plot_averages_by_distance(df, 'all-deprel-dist-to-en-ave-huelayer.png', fignum, use_en_source=False, num_accs=8*3, pointplot=True, hue='Layer')




############ Semantic Dependencies #####################
print('plotting semantic dependencies')

dm_maj, dm_mfl = 0.39297616*100, 0.6857216098*100
pas_maj, pas_mfl = 0.1344005122*100, 0.6111093325*100
psd_maj, psd_mfl = 0.1903355445*100, 0.4679223357*100
majs_dm, mfls_dm = [dm_maj]*6, [dm_mfl]*6
majs_pas, mfls_pas = [pas_maj]*6, [pas_mfl]*6
majs_psd, mfls_psd = [psd_maj]*6, [psd_mfl]*6
majs_dmpaspsd, mfls_dmpaspsd = [majs_dm, majs_pas, majs_psd], [mfls_dm, mfls_pas, mfls_psd]
data_file = os.path.join(RESULTS_DIR, 'results-un-semdeprel-dm-plot.txt')
df = load_data(data_file)
print(fignum)
#fignum = plot_pairs_by_layer(df, pairs_en, majs_dm, mfls_dm, 'all-semdeprel-dm.png', fignum, ymin=65, plot_maj=False, 
#                             legend=False, xsubs=3, ysubs=2, scale_figx=1.5, scale_figy=1.1, delta_val=2, suptitle='dm')
fignum = plot_averages(df, 'all-semdeprel-dm-en-to-ave.png', fignum, use_en_source=True)

data_file = os.path.join(RESULTS_DIR, 'results-un-semdeprel-pas-plot.txt')
df = load_data(data_file)
print(fignum)
#fignum = plot_pairs_by_layer(df, pairs_en, majs_pas, mfls_pas, 'all-semdeprel-pas.png', fignum, ymin=55, plot_maj=False, 
#                             legend=False, xsubs=3, ysubs=2, scale_figx=1.5, scale_figy=1.1, delta_val=2, suptitle='pas')
fignum = plot_averages(df, 'all-semdeprel-pas-en-to-ave.png', fignum, use_en_source=True)

data_file = os.path.join(RESULTS_DIR, 'results-un-semdeprel-psd-plot.txt')
df = load_data(data_file)
print(fignum)
#fignum = plot_pairs_by_layer(df, pairs_en, majs_psd, mfls_psd, 'all-semdeprel-psd.png', fignum, ymin=40, plot_maj=False, 
#                             legend=False, xsubs=3, ysubs=2, scale_figx=1.5, scale_figy=1.1, delta_val=2, suptitle='psd')
fignum = plot_averages(df, 'all-semdeprel-psd-en-to-ave.png', fignum, use_en_source=True)

data_file = os.path.join(RESULTS_DIR, 'results-un-semdeprel-dmpaspsd-plot.txt')
df = load_data(data_file)
print(fignum)
#fignum = plot_pairs_by_layer_semdeprel_schemes(df, [pairs_en, pairs_en, pairs_en], majs_dmpaspsd, mfls_dmpaspsd, 
#                                               'all-semdeprel-dmpaspsd.png', fignum, ymin=40, ymax=100, plot_maj=False)
#fignum = plot_pairs_by_layer_semdeprel_schemes2(df, [pairs_en, pairs_en, pairs_en], majs_dmpaspsd, mfls_dmpaspsd, 
#                                               'all-semdeprel-dmpaspsd2.png', fignum, ymin=40, ymax=100, plot_maj=False)


proportions_en_semdeprel_dm = [0.3988379152, 0.2465527708, 0.0140057237, 0.111395369, 0.02272135981, 0.004162691874, 0.0100598387, 0.1192437776, 0.005376810337, 0.01491631255, 0.00893244298, 0.007024542538, 0.0248894285, 0.004249414621, 0.007631601769]
source2proportions_semdeprel_dm_min100 = {'en': proportions_en_semdeprel_dm}
types_semdeprel_dm_min100_file_pref = os.path.join(RESULTS_DIR, 'results-un-semdeprel-types-nonull-min100-')
types_min100_file_suf = '-plot.txt'

fignum += 1
print(fignum)
#fignum = plot_pairs_by_relation_type(types_semdeprel_dm_min100_file_pref, types_min100_file_suf, pairs_en, source2proportions_semdeprel_dm_min100, 'types-semdeprel-dm-min100', fignum)


types_dm_all_file = os.path.join(RESULTS_DIR, 'results-un-semdeprel-types-nonull-min100-plot.txt')
df = load_data_by_type_all(types_dm_all_file)
fignum = plot_averages_by_type(df, 'all-semdeprel-dm-types-ave.png', fignum, use_en_source=True, pointplot=True)
fignum = plot_averages_by_type(df, 'all-semdeprel-dm-types-ave-box.png', fignum, use_en_source=True, pointplot=False)
fignum = plot_averages_by_type(df, 'all-semdeprel-dm-types-no0-ave.png', fignum, use_en_source=True, pointplot=True, layer0=False)
fignum = plot_averages_by_type(df, 'all-semdeprel-dm-types-ave-no0-box.png', fignum, use_en_source=True, pointplot=False, layer0=False)

types_dm_all_deltas_file = os.path.join(RESULTS_DIR, 'results-un-semdeprel-types-nonull-min100-deltas-plot.txt')
df = load_data_by_type_all(types_dm_all_deltas_file)
fignum = plot_averages_by_type(df, 'all-semdeprel-dm-types-deltas-ave.png', fignum, use_en_source=True, pointplot=True)
fignum = plot_averages_by_type(df, 'all-semdeprel-dm-types-deltas-ave-box.png', fignum, use_en_source=True, pointplot=False)




data_file_semdeprel_dm_dists = os.path.join(RESULTS_DIR, 'results-un-semdeprel-distance-nonull-plot.txt')
df = load_data_by_distance(data_file_semdeprel_dm_dists)
print(fignum)
#fignum = plot_pairs_by_relation_distance_wrt_layer(df, pairs_en, 'dists-pairs-semdeprel-dm-layer-nonull.png', fignum, subplot_dims=(3,2))
print(fignum)
#fignum = plot_pairs_by_relation_distance_wrt_distance(df, pairs_en, 'dists-pairs-semdeprel-dm-distance-nonull.png', fignum, subplot_dims=(3,2))
# num_accs = 8 distances * 3 runs
fignum = plot_averages_by_distance(df, 'all-semdeprel-dm-dist-en-to-ave-huedist.png', fignum, use_en_source=True, num_accs=8*3, pointplot=True, hue='Distance')
fignum = plot_averages_by_distance(df, 'all-semdeprel-dm-dist-en-to-ave-huelayer.png', fignum, use_en_source=True, num_accs=8*3, pointplot=True, hue='Layer')



proportions_en_semdeprel_pas = [0.1361175059, 0.008138516909, 0.008138516909, 0.03174345838, 0.03177588275, 0.02045977757, 0.00411789501, 0.01293732369, 0.008041243799, 0.02636101294, 0.02636101294, 0.09584643818, 0.1004831231, 0.008657306832, 0.009111248014, 0.09403067345, 0.09536007263, 0.0650757109, 0.006582147142, 0.1017476736, 0.101261308, 0.007652151357]
source2proportions_semdeprel_pas_min100 = {'en': proportions_en_semdeprel_pas}
types_semdeprel_pas_min100_file_pref = os.path.join(RESULTS_DIR, 'results-un-semdeprel-pas-types-nonull-min100-')
types_min100_file_suf = '-plot.txt'

print(fignum)
#fignum = plot_pairs_by_relation_type(types_semdeprel_pas_min100_file_pref, types_min100_file_suf, pairs_en, source2proportions_semdeprel_pas_min100, 'types-semdeprel-pas-min100', fignum)


types_pas_all_file = os.path.join(RESULTS_DIR, 'results-un-semdeprel-pas-types-nonull-min100-plot.txt')
df = load_data_by_type_all(types_pas_all_file)
fignum = plot_averages_by_type(df, 'all-semdeprel-pas-types-ave.png', fignum, use_en_source=True, pointplot=True)
fignum = plot_averages_by_type(df, 'all-semdeprel-pas-types-ave-box.png', fignum, use_en_source=True, pointplot=False)
fignum = plot_averages_by_type(df, 'all-semdeprel-pas-types-no0-ave.png', fignum, use_en_source=True, pointplot=True, layer0=False)
fignum = plot_averages_by_type(df, 'all-semdeprel-pas-types-ave-no0-box.png', fignum, use_en_source=True, pointplot=False, layer0=False)

types_pas_all_deltas_file = os.path.join(RESULTS_DIR, 'results-un-semdeprel-pas-types-nonull-min100-deltas-plot.txt')
df = load_data_by_type_all(types_pas_all_deltas_file)
fignum = plot_averages_by_type(df, 'all-semdeprel-pas-types-deltas-ave.png', fignum, use_en_source=True, pointplot=True)
fignum = plot_averages_by_type(df, 'all-semdeprel-pas-types-deltas-ave-box.png', fignum, use_en_source=True, pointplot=False)



data_file_semdeprel_pas_dists = os.path.join(RESULTS_DIR, 'results-un-semdeprel-pas-distance-nonull-plot.txt')
df = load_data_by_distance(data_file_semdeprel_pas_dists)
print(fignum)
#fignum = plot_pairs_by_relation_distance_wrt_layer(df, pairs_en, 'dists-pairs-semdeprel-pas-layer-nonull.png', fignum, subplot_dims=(3,2))
print(fignum)
#fignum = plot_pairs_by_relation_distance_wrt_distance(df, pairs_en, 'dists-pairs-semdeprel-pas-distance-nonull.png', fignum, subplot_dims=(3,2))
fignum = plot_averages_by_distance(df, 'all-semdeprel-pas-dist-en-to-ave-huedist.png', fignum, use_en_source=True, num_accs=8*3, pointplot=True, hue='Distance')
fignum = plot_averages_by_distance(df, 'all-semdeprel-pas-dist-en-to-ave-huelayer.png', fignum, use_en_source=True, num_accs=8*3, pointplot=True, hue='Layer')




proportions_en_semdeprel_psd = [0.006032554995, 0.1821727599, 0.01118102865, 0.006292578917, 0.007280669822, 0.04597222945, 0.02901866972, 0.004992459306, 0.004940454522, 0.06547402361, 0.007436684175, 0.006708617193, 0.006188569348, 0.005252483228, 0.03094284674, 0.02194601903, 0.02797857403, 0.009048832493, 0.006552602839, 0.04987258828, 0.1682874824, 0.01050496646, 0.02548234438, 0.01601747361, 0.2044308076, 0.005668521504, 0.03432315773]
source2proportions_semdeprel_psd_min100 = {'en': proportions_en_semdeprel_psd}
types_semdeprel_psd_min100_file_pref = os.path.join(RESULTS_DIR, 'results-un-semdeprel-psd-types-nonull-min100-')
types_min100_file_suf = '-plot.txt'

print(fignum)
#fignum = plot_pairs_by_relation_type(types_semdeprel_psd_min100_file_pref, types_min100_file_suf, pairs_en, source2proportions_semdeprel_psd_min100, 'types-semdeprel-psd-min100', fignum)


types_psd_all_file = os.path.join(RESULTS_DIR, 'results-un-semdeprel-psd-types-nonull-min100-plot.txt')
df = load_data_by_type_all(types_psd_all_file)
fignum = plot_averages_by_type(df, 'all-semdeprel-psd-types-ave.png', fignum, use_en_source=True, pointplot=True)
fignum = plot_averages_by_type(df, 'all-semdeprel-psd-types-ave-box.png', fignum, use_en_source=True, pointplot=False)
fignum = plot_averages_by_type(df, 'all-semdeprel-psd-types-no0-ave.png', fignum, use_en_source=True, pointplot=True, layer0=False)
fignum = plot_averages_by_type(df, 'all-semdeprel-psd-types-ave-no0-box.png', fignum, use_en_source=True, pointplot=False, layer0=False)

types_psd_all_deltas_file = os.path.join(RESULTS_DIR, 'results-un-semdeprel-psd-types-nonull-min100-deltas-plot.txt')
df = load_data_by_type_all(types_psd_all_deltas_file)
fignum = plot_averages_by_type(df, 'all-semdeprel-psd-types-deltas-ave.png', fignum, use_en_source=True, pointplot=True)
fignum = plot_averages_by_type(df, 'all-semdeprel-psd-types-deltas-ave-box.png', fignum, use_en_source=True, pointplot=False)




data_file_semdeprel_psd_dists = os.path.join(RESULTS_DIR, 'results-un-semdeprel-psd-distance-nonull-plot.txt')
df = load_data_by_distance(data_file_semdeprel_psd_dists)
print(fignum)
#fignum = plot_pairs_by_relation_distance_wrt_layer(df, pairs_en, 'dists-pairs-semdeprel-psd-layer-nonull.png', fignum, subplot_dims=(3,2))
print(fignum)
#fignum = plot_pairs_by_relation_distance_wrt_distance(df, pairs_en, 'dists-pairs-semdeprel-psd-distance-nonull.png', fignum, subplot_dims=(3,2))
fignum = plot_averages_by_distance(df, 'all-semdeprel-psd-dist-en-to-ave-huedist.png', fignum, use_en_source=True, num_accs=8*3, pointplot=True, hue='Distance')
fignum = plot_averages_by_distance(df, 'all-semdeprel-psd-dist-en-to-ave-huelayer.png', fignum, use_en_source=True, num_accs=8*3, pointplot=True, hue='Layer')






################# Morphology #############################
en_maj_morph, en_mfl_morph = 0.1285862289*100, 0.8146318138*100
ar_maj_morph, ar_mfl_morph = 0.1366402491*100, 0.7447636569*100
es_maj_morph, es_mfl_morph = 0.1573333333*100, 0.86175*100
fr_maj_morph, fr_mfl_morph = 0.1487025948*100, 0.8479041916*100
ru_maj_morph, ru_mfl_morph = 0.188344302*100, 0.6339625909*100
zh_maj_morph, zh_mfl_morph = 0.2753080253*100, 0.8125208125*100
majs_morph = [en_maj_morph]*6 + [ar_maj_morph, es_maj_morph, fr_maj_morph, ru_maj_morph, zh_maj_morph]
mfls_morph = [en_mfl_morph]*6 + [ar_mfl_morph, es_mfl_morph, fr_mfl_morph, ru_mfl_morph, zh_mfl_morph]



data_file = os.path.join(RESULTS_DIR, 'results-un-morph-plot.txt')
df = load_data(data_file)
#fignum = plot_pairs_by_layer(df, pairs, majs_morph, mfls_morph, 'all-morph.png', fignum, ymin=60, plot_maj=False)

# averages
fignum += 1
fignum = plot_averages(df, 'all-morph-en-to-ave.png', fignum, use_en_source=True)
fignum = plot_averages(df, 'all-morph-to-en-ave.png', fignum, use_en_source=False)




