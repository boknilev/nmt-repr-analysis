# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:44:00 2016

@author: belinkov
"""

from matplotlib import pyplot as plt
import numpy as np


### effect of representation ###
# ar-he, uni, 2lstm500, after layer 2

# accuracies: unseen, seen, all
labels = ['Unseen', 'Seen', 'All']
groups = ['POS', 'Morphology']
sets = ['Word', 'Char']
word_pos_accs = np.array([37.93, 81.69, 78.20])
char_pos_accs = np.array([75.51, 93.95, 92.48])
word_morph_accs = np.array([17.21, 69.36, 65.20])
char_morph_accs = np.array([49.89, 82.24, 79.66])

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                #'%d' % int(height),
                 '{}'.format(np.round(height, 1)),
                ha='center', va='bottom')

def plot_bars_two_sets(accs1, accs2, sets, labels, title, fignum, filename, auto_label=True):
    """ bar plot comparing two sets of results """
    
    assert len(accs1) == len(accs2) and len(accs2) == len(labels), 'incompatible arguments in plot_bars_two_sets'
    
    plt.figure(fignum)
    ind = np.arange(len(labels))
    width = 0.35
    
    rects1 = plt.bar(ind, accs1, width, color='r', hatch='/', label=sets[0])
    rects2 = plt.bar(ind + width, accs2, width, color='y', hatch='\\', label=sets[1])
    
    plt.ylabel('Accuracy', size='large', fontweight='demibold')
    plt.title(title, fontweight='demibold')
    plt.xticks(ind + width, labels, size='large', fontweight='demibold')
    plt.legend(loc='upper left', prop={'size':12})
    #plt.ylim([30,100])
    
    if auto_label:
        autolabel(rects1)
        autolabel(rects2)
    
    #plt.show()
    plt.savefig(filename)
    
    
def plot_bars_two_sets_stacked(word_pos_accs, char_pos_accs, word_morph_accs, char_morph_accs,  sets, labels, title, fignum, filename):
    """ bar plot comparing two sets of results """
    
    assert len(word_pos_accs) == len(char_pos_accs) and len(char_pos_accs) == len(word_morph_accs) and len(word_morph_accs) == len(char_morph_accs), 'incompatible arguments in plot_bars_two_sets'
    
    plt.figure(fignum)
    ind = np.arange(len(labels))
    width = 0.35
    
    rects1 = plt.bar(ind, word_pos_accs, width, color='r', hatch='/', label=sets[0])
    rects2 = plt.bar(ind, char_pos_accs-word_pos_accs, width, bottom=word_pos_accs, color='y', hatch='/', label=sets[1])
    rects3 = plt.bar(ind + width, word_morph_accs, width, color='r', hatch='\\', label=sets[2])
    rects4 = plt.bar(ind + width, char_morph_accs-word_morph_accs, width, bottom=word_morph_accs, color='y', hatch='\\', label=sets[3])
    
    plt.ylabel('Accuracy', size='large', fontweight='demibold')
    plt.title(title, fontweight='demibold')
    plt.xticks(ind + width, labels, size='large', fontweight='demibold')
    plt.legend(loc='upper left', prop={'size':12})
    #plt.ylim([30,100])
    

    #plt.show()
    plt.savefig(filename)    
    

def plot_bars_two_sets_ratios(word_pos_accs, char_pos_accs, word_morph_accs, char_morph_accs,  sets, labels, title, fignum, filename):
    """ bar plot comparing two sets of results """
    
    assert len(word_pos_accs) == len(char_pos_accs) and len(char_pos_accs) == len(word_morph_accs) and len(word_morph_accs) == len(char_morph_accs), 'incompatible arguments in plot_bars_two_sets'
    
    plt.figure(fignum)
    ind = np.arange(len(labels))
    width = 0.35
    
    word_pos_errors = 100-np.array(word_pos_accs, dtype='float')
    char_pos_errors = 100-np.array(char_pos_accs, dtype='float')
    word_morph_errors = 100-np.array(word_morph_accs, dtype='float')
    char_morph_errors = 100-np.array(char_morph_accs, dtype='float')
    word_char_pos_error_reduction = (word_pos_errors - char_pos_errors) / word_pos_errors
    word_char_morph_error_reduction = (word_morph_errors - char_morph_errors) / word_morph_errors
    word_char_pos_absolute_acc_difference = np.array(char_pos_accs, dtype='float') - np.array(word_pos_accs, dtype='float')
    word_char_morph_absolute_acc_difference = np.array(char_morph_accs, dtype='float') - np.array(word_morph_accs, dtype='float')
    
    rects1 = plt.bar(ind, word_char_pos_absolute_acc_difference, width, color='g', hatch='/', label=sets[0])
    rects3 = plt.bar(ind + width, word_char_morph_absolute_acc_difference, width, color='c', hatch='\\', label=sets[1])
    
    plt.ylabel('Improvement in Accuracy', size='large', fontweight='demibold')
    plt.title(title, fontweight='demibold')
    plt.xticks(ind + width/2, labels, size='large', fontweight='demibold')
    plt.legend(loc='upper right', prop={'size':14})
    #plt.ylim([30,100])
    plt.tight_layout()
    

    #plt.show()
    plt.savefig(filename)    
    

    
#plot_bars_two_sets(word_pos_accs, char_pos_accs, sets, labels, 'POS Accuracy by Representation Type', 1, 'pos-acc-repr-type.png', auto_label=False)
#plot_bars_two_sets(word_morph_accs, char_morph_accs, sets, labels, 'Morphology Accuracy by Representation Type', 2, 'morph-acc-repr-type.png', auto_label=False)
sets_stacked = ['Word POS', 'Char POS', 'Word Morph', 'Char Morph']
#plot_bars_two_sets_stacked(word_pos_accs, char_pos_accs, word_morph_accs, char_morph_accs, sets_stacked, labels, 'Accuracy by Representation Type', 1, 'acc-repr-type-stacked.png')
#plot_bars_two_sets_ratios(word_pos_accs, char_pos_accs, word_morph_accs, char_morph_accs, ['POS', 'Morph'], labels, 'Improvement in POS and Morphology Accuracy', 111, 'acc-repr-type-diff.png')



### effect of layer depth ###
# ar-he, uni, 2lstm500
word_pos_layer2_acc = 78.20
word_pos_layer1_acc = 79.4
word_pos_layer0_acc = 77.25
word_morph_layer2_acc = 65.20
word_morph_layer1_acc = 67.03
word_morph_layer0_acc = 63.75
char_pos_layer2_acc = 92.48
char_pos_layer1_acc = 94.05
char_pos_layer0_acc = 93.27
char_morph_layer2_acc = 79.66
char_morph_layer1_acc = 81.06
char_morph_layer0_acc = 76.86
word_layer2_bleu = 9.91
word_layer1_bleu = 8.80
char_layer2_bleu = 10.65
char_layer1_bleu = 10.09

# layer2 - layer1
word_pos_diff = word_pos_layer2_acc - word_pos_layer1_acc
word_morph_diff = word_morph_layer2_acc - word_morph_layer1_acc
word_bleu_diff = word_layer2_bleu - word_layer1_bleu
char_pos_diff = char_pos_layer2_acc - char_pos_layer1_acc
char_morph_diff = char_morph_layer2_acc - char_morph_layer1_acc
char_bleu_diff = char_layer2_bleu - char_layer1_bleu
word_diffs = [word_pos_diff, word_morph_diff, word_bleu_diff]
char_diffs = [char_pos_diff, char_morph_diff, char_bleu_diff]

word_pos_accs = [word_pos_layer0_acc, word_pos_layer1_acc, word_pos_layer2_acc]
word_morph_accs = [word_morph_layer0_acc, word_morph_layer1_acc, word_morph_layer2_acc]
char_pos_accs = [char_pos_layer0_acc, char_pos_layer1_acc, char_pos_layer2_acc]
char_morph_accs = [char_morph_layer0_acc, char_morph_layer1_acc, char_morph_layer2_acc]
layer_accs = [word_pos_accs, word_morph_accs, char_pos_accs, char_morph_accs]
layer_labels = ['Word-POS', 'Word-Morph', 'Char-POS', 'Char-Morph']
layer_colors = ['r', 'y', 'g', 'c']
layer_markers = ['o', 's', 'P', '*']


# effect of layer depth in different languages
ar_en_word_layer0, ar_en_word_layer1, ar_en_word_layer2 = 77.09, 81.07, 80.31
ar_he_word_layer0, ar_he_word_layer1, ar_he_word_layer2 = 77.25, 79.40, 78.20
de_en_word_layer0, de_en_word_layer1, de_en_word_layer2 = 91.14, 93.57, 93.54
fr_en_word_layer0, fr_en_word_layer1, fr_en_word_layer2 = 92.08, 95.06, 94.61
cz_en_word_layer0, cz_en_word_layer1, cz_en_word_layer2 = 76.26, 76.98, 75.71
word_layer0_all_langs = [ar_en_word_layer0, ar_he_word_layer0, de_en_word_layer0, fr_en_word_layer0, cz_en_word_layer0]
word_layer1_all_langs = [ar_en_word_layer1, ar_he_word_layer1, de_en_word_layer1, fr_en_word_layer1, cz_en_word_layer1]
word_layer2_all_langs = [ar_en_word_layer2, ar_he_word_layer2, de_en_word_layer2, fr_en_word_layer2, cz_en_word_layer2]
ar_en_word_layers = [ar_en_word_layer0, ar_en_word_layer1, ar_en_word_layer2]
ar_he_word_layers = [ar_he_word_layer0, ar_he_word_layer1, ar_he_word_layer2]
de_en_word_layers = [de_en_word_layer0, de_en_word_layer1, de_en_word_layer2]
fr_en_word_layers = [fr_en_word_layer0, fr_en_word_layer1, fr_en_word_layer2]
cz_en_word_layers = [cz_en_word_layer0, cz_en_word_layer1, cz_en_word_layer2]
layer_labels = ['Layer 0', 'Layer 1', 'Layer 2']
layer_sets = ['Ar-En', 'Ar-He', 'De-En', 'Fr-En', 'Cz-En']

def plot_bars_layer_all_langs(accs1, accs2, accs3, sets, labels, title, fignum, filename, indices=None, legend_loc=None, opacity=1.0):
    
    assert len(accs1) == len(accs2) and len(accs2) == len(accs3) and len(accs3) == len(labels), 'incompatible arguments in plot_bars_four_sets'
    
    plt.figure(fignum)
    ind = indices
    if ind == None:
        ind = np.arange(len(labels))
    #print ind
    width = 0.25
    
    rects1 = plt.bar(ind, accs1, width, color='y', hatch='/', label=sets[0], alpha=opacity)
    rects2 = plt.bar(ind + width, accs2, width, color='r', hatch='+', label=sets[1], alpha=opacity)
    rects3 = plt.bar(ind + width + width, accs3, width, color='c', hatch='\\', label=sets[2], alpha=opacity)
    
    
    plt.ylabel('Accuracy', size='large', fontweight='demibold')
    plt.title(title, fontweight='demibold')
    plt.xticks(ind + width, labels, size='large', fontweight='demibold')
    loc = legend_loc
    if loc == None:
        loc = 'upper left'
    plt.legend(loc=loc, prop={'size':14})
    plt.ylim([60,100])
    

    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)
    
    plt.tight_layout()    
    #plt.show()    
    plt.savefig(filename)    

plot_bars_layer_all_langs(word_layer0_all_langs, word_layer1_all_langs, word_layer2_all_langs, layer_labels, layer_sets, \
                           'POS Accuracy by Representation Layer', 3333, 'rep-layer-acc-all-langs.png')
    

def plot_bars_two_groups(group1, group2, groups, labels, title, fignum, filename):
    
    assert len(group1) == len(group2) and len(group2) == len(labels), 'incompatible arguments in plot_bars_two_groups'
    
    plt.figure(fignum)
    ind = np.arange(len(labels))
    width = 1.0
    
    rects1 = plt.bar(ind, group1, width, color='r', hatch='/', label=groups[0])
    rects2 = plt.bar(ind+len(labels)+width, group2, width, color='y', hatch='\\', label=groups[1])
    
    plt.ylabel('Change in Accuracy or BLEU', size='large', fontweight='demibold')
    plt.title(title, fontweight='demibold')
    ticks = np.concatenate((ind + width/2, len(labels) + ind + width + width/2))
    #print ticks
    plt.xticks(ticks, labels + labels, size='large')
    plt.axhline(color='black')
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(filename)

#plot_bars_two_groups(word_diffs, char_diffs, ['Word', 'Char'], ['POS', 'Morph', 'BLEU'], 'Effect of Representation Layer', 3, 'layer-effect.png')

def plot_lines(xs, accs, labels, colors, markers, title, fignum, filename):

    plt.figure(fignum)
    for i in xrange(len(accs)):
        plt.plot(xs, accs[i], '--' + markers[i], color=colors[i], label=labels[i], lw=2, markersize=10)
    plt.title(title, fontweight='demibold')
    plt.xlabel('Layer', size='large', fontweight='demibold')
    plt.ylabel('Accuracy', size='large', fontweight='demibold')
    plt.xticks([0, 1, 2])
    plt.ylim(60,100)
    plt.legend(loc=(0.02,0.5), prop={'weight':'medium'})
    plt.tight_layout()
    plt.savefig(filename)

#plot_lines([0, 1, 2], layer_accs, layer_labels, layer_colors, layer_markers, 'Accuracy by Representation Layer', 33, 'rep-layer-acc-lines.png')


def plot_bars_five_sets(accs1, accs2, accs3, accs4, accs5, sets, labels, title, fignum, filename, indices=None, legend_loc=None, opacity=1.0):
    """ bar plot comparing four sets of results """
    
    assert len(accs1) == len(accs2) and len(accs2) == len(accs3) and len(accs3) == len(accs4) and len(accs4) == len(accs5) and len(accs5) == len(labels), 'incompatible arguments in plot_bars_four_sets'
    
    plt.figure(fignum)
    ind = indices
    if ind == None:
        ind = np.arange(len(labels))
    #print ind
    width = 0.2
    
    rects1 = plt.bar(ind, accs1, width, color='r', hatch='/', label=sets[0], alpha=opacity)
    rects2 = plt.bar(ind + width, accs2, width, color='y', hatch='O', label=sets[1], alpha=opacity)
    rects3 = plt.bar(ind + width + width, accs3, width, color='g', hatch='+', label=sets[2], alpha=opacity)
    rects4 = plt.bar(ind + width + width + width, accs4, width, color='c', hatch='\\', label=sets[3], alpha=opacity)
    rects5 = plt.bar(ind + width + width + width + width, accs5, width, color='b', hatch='*', label=sets[4], alpha=opacity)
    
    
    plt.ylabel('Accuracy', size='large', fontweight='demibold')
    plt.title(title, fontweight='demibold')
    plt.xticks(ind + width + width, labels, size='large', fontweight='demibold')
    loc = legend_loc
    if loc == None:
        loc = 'upper left'
    plt.legend(loc=loc, prop={'size':12})
    #plt.ylim([40,95])
    

    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)
    
    plt.tight_layout()    
    #plt.show()    
    plt.savefig(filename)

#plot_bars_five_sets(ar_en_word_layers, ar_he_word_layers, de_en_word_layers, fr_en_word_layers, cz_en_word_layers, layer_sets, layer_labels, 'POS Accuracy by Representation Layer', 333, 'rep-layer-acc-all-langs.png')


### effect of target language ###
# uni, 2lstm500
word_pos_he, word_pos_en, word_pos_ar, word_pos_de = 78.13, 80.21, 67.21, 78.85
char_pos_he, char_pos_en, char_pos_ar, char_pos_de = 92.67, 93.63, 87.72, 93.05
word_morph_he, word_morph_en, word_morph_ar, word_morph_de = 64.87, 67.18, 55.63, 65.91
char_morph_he, char_morph_en, char_morph_ar, char_morph_de = 80.5, 81.49, 75.21, 80.61
bleu_word_he, bleu_char_he = 9.51, 11.15
bleu_word_en_1, bleu_word_en_2, bleu_char_en_1, bleu_char_en_2 = 24.72, 22.88, 29.46, 26.18
bleu_word_en, bleu_char_en = np.mean([bleu_word_en_1, bleu_word_en_2]), np.mean([bleu_char_en_1, bleu_char_en_2])
bleu_word_ar, bleu_char_ar = 80.43, 75.48
bleu_word_de, bleu_char_de = 11.49, 12.86
labels = ['POS', 'BLEU']
sets = ['Ar', 'He', 'En']
word_pos_accs = [word_pos_ar, word_pos_he, word_pos_en]
word_pos_accs_all = [word_pos_ar, word_pos_he, word_pos_en, word_pos_de]
word_bleus = [bleu_word_ar, bleu_word_he, bleu_word_en]
word_bleus_all = [bleu_word_ar, bleu_word_he, bleu_word_en, bleu_word_de]
ar_word_pos_bleu = [word_pos_ar, bleu_word_ar]
en_word_pos_bleu = [word_pos_en, bleu_word_en]
he_word_pos_bleu = [word_pos_he, bleu_word_he]
de_word_pos_bleu = [word_pos_de, bleu_word_de]
ar_pos_bleu = [word_pos_ar, bleu_word_ar, char_pos_ar, bleu_char_ar]
en_pos_bleu = [word_pos_en, bleu_word_en, char_pos_en, bleu_char_en]
he_pos_bleu = [word_pos_he, bleu_word_he, char_pos_he, bleu_char_he]
de_pos_bleu = [word_pos_de, bleu_word_de, char_pos_de, bleu_char_de]
labels2 = ['Word POS', 'Word BLEU', 'Char POS', 'Char BLEU']
ar_word_morph_bleu = [word_morph_ar, bleu_word_ar]
en_word_morph_bleu = [word_morph_en, bleu_word_en]
he_word_morph_bleu = [word_morph_he, bleu_word_he]
ar_morph_bleu = [word_morph_ar, bleu_word_ar, char_morph_ar, bleu_char_ar]
en_morph_bleu = [word_morph_en, bleu_word_en, char_morph_en, bleu_char_en]
he_morph_bleu = [word_morph_he, bleu_word_he, char_morph_he, bleu_char_he]
labels3 = ['Word Morph', 'Word BLEU', 'Char Morph', 'Char BLEU']
ar_word_pos_morph_bleu = [word_pos_ar, word_morph_ar, bleu_word_ar]
en_word_pos_morph_bleu = [word_pos_en, word_morph_en, bleu_word_en]
he_word_pos_morph_bleu = [word_pos_he, word_morph_he, bleu_word_he]
de_word_pos_morph_bleu = [word_pos_de, word_morph_de, bleu_word_de]


def plot_bars_three_sets(accs1, accs2, accs3, sets, labels, title, fignum, filename, indices=None, legend_loc=None, ylabel='Accuracy or BLEU'):
    """ bar plot comparing three sets of results """
    
    assert len(accs1) == len(accs2) and len(accs2) == len(accs3) and len(accs3) == len(labels), 'incompatible arguments in plot_bars_three_sets'
    
    plt.figure(fignum)
    ind = indices
    if ind == None:
        ind = np.arange(len(labels))
    print ind
    width = 0.25
    
    rects1 = plt.bar(ind, accs1, width, color='r', hatch='/', label=sets[0])
    rects2 = plt.bar(ind + width, accs2, width, color='y', hatch='O', label=sets[1])
    rects3 = plt.bar(ind + width + width, accs3, width, color='g', hatch='+', label=sets[2])
    
    plt.ylabel(ylabel, size='large', fontweight='demibold')
    plt.title(title, fontweight='demibold')
    plt.xticks(ind + width + width/2, labels, size='large', fontweight='demibold')
    loc = legend_loc
    if loc == None:
        loc = 'upper left'
    plt.legend(loc=loc, prop={'size':12})
    #plt.ylim([30,100])
    

    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)
    
    plt.tight_layout()    
    #plt.show()    
    plt.savefig(filename)


#plot_bars_three_sets(ar_word_pos_bleu, he_word_pos_bleu, en_word_pos_bleu, sets, labels, 'Effect of Target Language on POS Accuracy', 4, 'pos-acc-target-lang.png')
#plot_bars_three_sets(ar_pos_bleu, he_pos_bleu, en_pos_bleu, sets, labels2, 'Effect of Target Language on POS Accuracy', 5, 'pos-acc-target-lang2.png', indices=np.array([0,1,2.5,3.5]), legend_loc=(0.33,0.75))
#plot_bars_three_sets(ar_morph_bleu, he_morph_bleu, en_morph_bleu, sets, labels3, 'Effect of Target Language on Morph Accuracy', 6, 'morph-acc-target-lang2.png', indices=np.array([0,1,2.5,3.5]), legend_loc=(0.33,0.75))



### effect of lstm unit ###
word_pos_hidden_acc = 78.20
word_pos_cell_acc = 78.19
char_pos_hidden_acc = 92.48
char_pos_cell_acc = 90.27
word_morph_hidden_acc = 65.20
word_morph_cell_acc = 65.20 ### TODO verify
char_morph_hidden_acc = 79.66
char_morph_cell_acc = 78.15



### Semtags ###
semtags_word_layer2 = [49.26, 86.86, 86.29]
semtags_char_layer2 = [68.63, 89.16, 88.85]
semtags_word_layer1 = [48.34, 87.42, 86.83]
semtags_char_layer1 = [62.24, 89.51, 89.09]
sets = ['Word, layer 2', 'Char, layer 2', 'Word, layer 1', 'Char, layer 1']
labels = ['Unseen', 'Seen', 'All']


def plot_bars_four_sets(accs1, accs2, accs3, accs4, sets, labels, title, fignum, filename, indices=None, legend_loc=None, opacity=1.0):
    """ bar plot comparing four sets of results """
    
    assert len(accs1) == len(accs2) and len(accs2) == len(accs3) and len(accs3) == len(accs4) and len(accs4) == len(labels), 'incompatible arguments in plot_bars_four_sets'
    
    plt.figure(fignum)
    ind = indices
    if ind == None:
        ind = np.arange(len(labels))
    #print ind
    width = 0.2
    
    rects1 = plt.bar(ind, accs1, width, color='r', hatch='/', label=sets[0], alpha=opacity)
    rects2 = plt.bar(ind + width, accs2, width, color='y', hatch='O', label=sets[1], alpha=opacity)
    rects3 = plt.bar(ind + width + width, accs3, width, color='g', hatch='+', label=sets[2], alpha=opacity)
    rects4 = plt.bar(ind + width + width + width, accs4, width, color='c', hatch='\\', label=sets[3], alpha=opacity)
    
    
    plt.ylabel('Accuracy or BLEU', size='large', fontweight='demibold')
    plt.title(title, fontweight='demibold')
    plt.xticks(ind + width + width/2, labels, size='large', fontweight='demibold')
    loc = legend_loc
    if loc == None:
        loc = 'upper left'
    plt.legend(loc=loc, prop={'size':14})
    #plt.ylim([40,95])
    

    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)
    
    plt.tight_layout()    
    #plt.show()    
    plt.savefig(filename)


#plot_bars_four_sets(semtags_word_layer2, semtags_char_layer2, semtags_word_layer1, semtags_char_layer1, sets, labels, 'Semtag Accuracy by Representation Type and Layer', 7, 'semtag-acc-type-layer.png', legend_loc='upper left')

# more on target language
labels = ['POS', 'BLEU']
sets_all = ['Ar', 'He', 'De', 'En']
#plot_bars_four_sets(ar_word_pos_bleu, he_word_pos_bleu, de_word_pos_bleu, en_word_pos_bleu, sets_all, labels, 'Effect of Target Language on POS Accuracy', 44, 'pos-acc-target-lang-all.png', legend_loc='upper right')
labels_pos_morph_bleu = ['POS', 'Morphology', 'BLEU']
plot_bars_four_sets(ar_word_pos_morph_bleu, he_word_pos_morph_bleu, de_word_pos_morph_bleu, en_word_pos_morph_bleu, sets_all, labels_pos_morph_bleu, 'Effect of Target Language on POS/Morph Accuracy', 44, 'pos-morph-acc-target-lang-all.png', legend_loc='upper right')


# order is: word layer 1, char layer 1, word layer 2, char layer 2
semtags_unseen = [semtags_word_layer1[0], semtags_char_layer1[0], semtags_word_layer2[0], semtags_char_layer2[0]]
semtags_seen = [semtags_word_layer1[1], semtags_char_layer1[1], semtags_word_layer2[1], semtags_char_layer2[1]]
semtags_all = [semtags_word_layer1[2], semtags_char_layer1[2], semtags_word_layer2[2], semtags_char_layer2[2]]
sets = ['Word, L1', 'Char, L1', 'Word, L2', 'Char, L2']
labels = ['Unseen', 'Seen', 'All']

def plot_bars_subplots_four_sets_three_groups(accs1, accs2, accs3, sets, labels, title, fignum, filename, indices=None, legend_loc=None):
    """ bar plot comparing four sets of results in 3 subplots """
    
    #assert len(accs1) == len(accs2) and len(accs2) == len(accs3) and len(accs3) == len(labels), 'incompatible arguments in plot_bars_four_sets'
    
    fig = plt.figure(fignum)
    ind = indices
    if ind == None:
        ind = np.arange(len(sets))*0.5
    print ind
    width = 0.5
    
    ax1 = plt.subplot(1,3,1)    
    rects1 = plt.bar(ind[0], accs1[0], width, color='r', hatch='/', label=sets[0])
    rects2 = plt.bar(ind[1], accs1[1], width, color='y', hatch='O', label=sets[1])
    rects3 = plt.bar(ind[2], accs1[2], width, color='g', hatch='+', label=sets[2])
    rects4 = plt.bar(ind[3], accs1[3], width, color='c', hatch='\\', label=sets[3])    
    plt.ylabel('Accuracy', size='large', fontweight='demibold')
    #plt.xlabel(labels[0],size='large', fontweight='demibold')
    plt.title(labels[0], fontweight='demibold')
    plt.xticks([0,0.5,1,1.5], ['' for x in sets], size='large', fontweight='demibold')
    plt.ylim([40,95])

    ax2 = plt.subplot(1,3,2)
    rects1 = plt.bar(ind[0], accs2[0], width, color='r', hatch='/', label=sets[0])
    rects2 = plt.bar(ind[1], accs2[1], width, color='y', hatch='O', label=sets[1])
    rects3 = plt.bar(ind[2], accs2[2], width, color='g', hatch='+', label=sets[2])
    rects4 = plt.bar(ind[3], accs2[3], width, color='c', hatch='\\', label=sets[3])    
    #plt.ylabel('Accuracy', size='large', fontweight='demibold')
    #plt.xlabel(labels[1], size='large', fontweight='demibold')
    plt.title(labels[1], fontweight='demibold')
    plt.xticks([0,0.5,1,1.5], ['' for x in sets], size='large', fontweight='demibold')
    plt.ylim([80,95])

    ax3 = plt.subplot(1,3,3)
    rects1 = plt.bar(ind[0], accs3[0], width, color='r', hatch='/', label=sets[0])
    rects2 = plt.bar(ind[1], accs3[1], width, color='y', hatch='O', label=sets[1])
    rects3 = plt.bar(ind[2], accs3[2], width, color='g', hatch='+', label=sets[2])
    rects4 = plt.bar(ind[3], accs3[3], width, color='c', hatch='\\', label=sets[3])    
    #plt.ylabel('Accuracy', size='large', fontweight='demibold')
    #plt.xlabel(labels[2], size='large', fontweight='demibold')
    plt.title(labels[2], fontweight='demibold')
    plt.xticks([0,0.5,1,1.5], ['' for x in sets], size='large', fontweight='demibold')
    plt.ylim([80,95])

    for ax in [ax1, ax2, ax3]:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    lgd = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4, fontsize=10)  
    

    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)

    #loc = legend_loc
    #if loc == None:
    #    loc = 'upper left'
    #plt.legend(loc=loc, prop={'size':9})
    #plt.legend()
    #plt.legend(loc = 'lower center', bbox_to_anchor = (0,0.5,1,1),
    #        bbox_transform = plt.gcf().transFigure )    

    sup = plt.suptitle('Semtag Accuracy', fontweight='demibold', size=16)
    
    #plt.tight_layout()    
    plt.subplots_adjust(top=0.86)
    #plt.show()    
    plt.savefig(filename, bbox_extra_artists=(lgd, sup), bbox_inches='tight')   
    


#plot_bars_subplots_four_sets_three_groups(semtags_unseen, semtags_seen, semtags_all, sets, labels, 'Semtag Accuracy', 8, 'semtag-acc.png')


