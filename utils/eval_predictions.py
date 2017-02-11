# -*- coding: utf-8 -*-
"""
Evaluate pos/morph predictions 

Usage: eval_predictions.py [--train_lbl_file train_lbl_file] [--train_file train_file] [--test_file test_file] 
                            [--fig_pref fig_pref] [--test_pred_file2 test_pred_file2] 
                            [--label1 label1] [--label2 label2] [--annotation annotation]
                            TEST_GOLD_FILE TEST_PRED_FILE

Arguments:
  TEST_GOLD_FILE  File with gold tags, once sentence per line
  TEST_PRED_FILE  File with predicted tags, one sentence per line
  
  
Options:
  -h, --help                           show this help message  
  --train_lbl_file train_lbl_file      train file with gold tags to collect frequency statistics
  --train_file train_file              train file with original text to collect statistics
  --test_file test_file                test file with original text (must be provided if train_file is given)
  --fig_pref fig_pref                  file prefix to save figures of TEST_PRED_FILE
  --test_pred_file2 test_pred_file2    file with predicted tags, one sentence per line, to compare with TEST_PRED_FILE
  --label1 label1                      string label for model 1 when comparing two predictions
  --label2 label2                      string label for model 2 when comparing two predictions
  --annotation annotation              annotation type: pos, morph (for plot title)
"""

from docopt import docopt
from sklearn import metrics 
import numpy as np
from itertools import izip
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import codecs, operator, sys



def get_predictions(gold_filename, pred_filename):
    """ Get gold and predicted tags
    
    return (gold, pred, tags): gold and pred are lists of tags
                                tags is a sorted list of unique tags
    """
    

    gold, pred = [], []
    tags = set()
    with open(gold_filename) as f_gold:
        with open(pred_filename) as f_pred:
            for gold_line, pred_line in izip(f_gold, f_pred):
                assert len(gold_line.split()) == len(pred_line.split()), 'incompatible lines:\ngold: ' + gold_line + 'pred: ' + pred_line
                for gold_tag, pred_tag in zip(gold_line.strip().split(), pred_line.strip().split()):
                    gold.append(gold_tag)
                    pred.append(pred_tag)
                    tags.add(gold_tag)
                    
    tags = sorted(tags)    
    return gold, pred, tags
    

def eval_predictions(gold, pred, tags, confusion_matrix_filename=None, fig_num=1):
    
    cm = metrics.confusion_matrix(gold, pred, tags)
    print 'confusion matrix:'
    print cm
    if confusion_matrix_filename:
        plt.figure(fig_num)
        plot_confusion_matrix(cm, tags, 'Confusion matrix', confusion_matrix_filename)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print 'normalized confusion matrix:'
    print cm_normalized
    if confusion_matrix_filename:
        plt.figure(fig_num+1)
        normalized_cm_filename = '.normalized.'.join(confusion_matrix_filename.rsplit('.', 1)) if confusion_matrix_filename else None
        plot_confusion_matrix(cm_normalized, tags, 'Normalized confusion matrix', 
                          normalized_cm_filename)
    report = metrics.classification_report(gold, pred, tags, digits=4)
    print 'classification report:'
    print report
    accuracy = metrics.accuracy_score(gold, pred)
    print 'accuracy:', accuracy
    f1_micro = metrics.f1_score(gold, pred, average='micro')
    f1_macro = metrics.f1_score(gold, pred, average='macro')
    f1_weighted = metrics.f1_score(gold, pred, average='weighted')
    print 'F1 micro:', f1_micro, 'F1 macro:', f1_macro, 'F1 weighted:', f1_weighted
    return fig_num+2
                

def plot_confusion_matrix(cm, tags, title, filename):
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(tags))
    plt.xticks(tick_marks, tags, rotation=90, size='x-small')
    plt.yticks(tick_marks, tags, size='x-small')
    #plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    if filename:
        plt.savefig(filename, bbox_inches='tight')


def filter_by_tag_freq(gold, pred, freq, freq_dict, op=operator.eq):
    
    filtered_gold, filtered_pred = [], []
    for g, p in zip(gold, pred):
        if op(freq_dict.get(g, 0), freq):
            filtered_gold.append(g)
            filtered_pred.append(p)
    return filtered_gold, filtered_pred


def filter_by_min_tag_freq(gold, pred, min_freq, freq_dict):
    
    filtered_gold, filtered_pred = [], []
    for g, p in zip(gold, pred):
        if freq_dict.get(g, 0) >= min_freq:
            filtered_gold.append(g)
            filtered_pred.append(p)
    return filtered_gold, filtered_pred
    

def filter_by_word_freq(gold, pred, words, freq, freq_dict, op=operator.eq):

    filtered_gold, filtered_pred = [], []
    for g, p, w in zip(gold, pred, words):
        if op(freq_dict.get(w, 0), freq):
            filtered_gold.append(g)
            filtered_pred.append(p)
    return filtered_gold, filtered_pred


def filter_by_min_word_freq(gold, pred, words, min_freq, freq_dict):

    filtered_gold, filtered_pred = [], []
    for g, p, w in zip(gold, pred, words):
        if freq_dict.get(w, 0) >= min_freq:
            filtered_gold.append(g)
            filtered_pred.append(p)
    return filtered_gold, filtered_pred
                

def get_freq_dict(filename, encoding='utf-8'):
    """ Get tag frequencies 
    
    filename: file with words/tags, one sentence per line 
    
    returns freq_dict: a dictionary from word/tag to frequency 
    """

    freq_dict = dict()    
    with codecs.open(filename, encoding=encoding) as f:
        for line in f:
            for t in line.strip().split():
                freq_dict[t] = freq_dict.get(t, 0) + 1
    return freq_dict


def eval_predictions_by_freq(gold, pred, tags, freq_dict, words=None, cumulative=False, min_freq=True, scale_acc=1.0):
    
    freqs, accuracies = [], []
    all_freqs = [0] + sorted(set(freq_dict.values()))
    #all_freqs = sorted(set(freq_dict.values()))
    if cumulative:
        comp_op = operator.ge if min_freq else operator.le
    else:
        comp_op = operator.eq
    for freq in all_freqs:
        if words:
            #filter_func = filter_by_min_word_freq if cumulative else filter_by_word_freq
            filtered_gold, filtered_pred = filter_by_word_freq(gold, pred, words, freq, freq_dict, op=comp_op)
        else:
            #filter_func = filter_by_min_tag_freq if cumulative else filter_by_tag_freq
            filtered_gold, filtered_pred = filter_by_tag_freq(gold, pred, freq, freq_dict, op=comp_op)
        if len(filtered_gold) == 0:
            continue
        accuracy = metrics.accuracy_score(filtered_gold, filtered_pred)
        if np.isnan(accuracy):
            print 'nan:', filtered_gold, filtered_pred
        accuracy *= scale_acc
        if cumulative:
            op_str = '>=' if min_freq else '<='
        else:
            op_str = '='
        
        print 'accuracy for train {} freq {} {}: {}'.format('word' if words else 'tag', op_str, freq, accuracy)
        freqs.append(freq)
        accuracies.append(accuracy)
        #print 'score report for train freq {}:'.format(freq)
        #print metrics.classification_report(filtered_gold, filtered_pred)
    return freqs, accuracies


def plot_accuracy_by_freq(freqs, accuracies, filename=None, title=''):
    
    plt.semilogx(freqs, accuracies, marker='.')
    plt.xlabel('Frequency')
    plt.ylabel('Accuracy')
    plt.title(title)
    if filename:
        print 'saving plot to:', filename
        plt.savefig(filename)


def plot_accuracy_by_freq_compare(freqs1, accuracies1, freqs2, accuracies2, label1, label2, title, filename=None, scale_acc=1.00, yscale_base=10.0, alpha=0.8, tags=None):
    
    plt.plot(freqs1, accuracies1, marker='o', color='r', label=label1, linestyle='None', fillstyle='none', alpha=alpha)
    plt.plot(freqs2, accuracies2, marker='+', color='c', label=label2, linestyle='None', fillstyle='none', alpha=alpha)

    if tags:
        print 'tags:', tags, 'len:', len(tags)
        print 'len(freqs1):', len(freqs1), 'len(freqs2)', len(freqs2)
        print 'len(accuracies1):', len(accuracies1), 'len(accuracies2)', len(accuracies2)
        if len(tags) == len(freqs1) and len(tags) == len(freqs2):
            print 'annotating tags'
            for i, tag in enumerate(tags):
                plt.annotate(tag, (freqs[1][i], accuracies[1][i]))


    plt.xscale('symlog')
    #plt.yscale('log', basey=yscale_base)
    plt.legend(loc='lower right', prop={'size':14})
    plt.xlabel('Frequency', size='large', fontweight='demibold')
    plt.ylabel('Accuracy', size='large', fontweight='demibold')
    plt.ylim(ymax=1.01*scale_acc)
    plt.title(title, fontweight='demibold')
    plt.tight_layout()
    if filename:
        print 'saving plot to:', filename
        plt.savefig(filename)    

def plot_accuracy_by_freq_compare_histogram(freqs1, accuracies1, freqs2, accuracies2, label1, label2, title, filename=None, scale_acc=1.00, yscale_base=10.0, alpha=0.5, tags=None):
    freqs1 = np.array(freqs1)
    freqs2 = np.array(freqs2)
    accuracies1 = np.array(accuracies1)
    accuracies2 = np.array(accuracies2)

    bins = ['0-5','5-10', '10-20', '20-50', '50-100', '100-500', '500-1000', '1000+']
    limits = [-1, 5, 10, 20, 50, 100, 500, 1000, 10000000]

    hist1 = [0]*(len(limits)-1)
    hist2 = [0]*(len(limits)-1)
    for idx,limit in enumerate(limits):
        if idx == 0:
            continue
        elem1 = np.where(np.logical_and(freqs1 > limits[idx-1], freqs1 <= limit))
        hist1[idx-1] = np.average(accuracies1[elem1])
        elem2 = np.where(np.logical_and(freqs2 > limits[idx-1], freqs2 <= limit))
        hist2[idx-1] = np.average(accuracies2[elem2])

    ind = np.arange(len(hist1))
    width = 0.35

    word_rects = plt.bar(ind, hist1, width, color='r', hatch='/')
    char_rects = plt.bar(ind+width, hist2, width, color='y', hatch='\\')

    
    plt.xticks(ind + width, tuple(bins))
    plt.legend((word_rects, char_rects), ('Word', 'Char'), loc='lower right')
    plt.ylabel('Average Bin Accuracy', size='large', fontweight='demibold')
    plt.xlabel('Frequency', size='large', fontweight='demibold')
    plt.ylim(ymax=1.01*scale_acc)
    plt.title(title, fontweight='demibold')
    if filename:
        print 'saving plot to:', filename
        plt.savefig(filename)

def eval_predictions_by_tag(gold, pred, tags):

    correct, wrong = np.zeros(len(tags)), np.zeros(len(tags))
    tag2idx = dict([(tag, i) for i, tag in enumerate(sorted(tags))])
    for g, p in zip(gold, pred):
        if g == p:
            correct[tag2idx[g]] += 1
        else:
            wrong[tag2idx[g]] += 1
    accs = 100.0*correct/(correct+wrong)
    return accs, sorted(tags)


def plot_accuracy_by_tag_compare(accuracies1, accuracies2, tags, tag_freq_dict, label1, label2, title, filename=None, scale_acc=1.00, yscale_base=10.0, alpha=0.5):
    #from adjustText import adjust_text 
    tag_freqs = [tag_freq_dict[tag] for tag in tags]
    #plt.plot(tag_freqs, accuracies1, marker='o', color='r', label=label1, linestyle='None', fillstyle='none', alpha=alpha)
    #plt.plot(tag_freqs, accuracies2, marker='+', color='y', label=label2, linestyle='None', fillstyle='none', alpha=alpha)
    # plt.plot(tag_freqs, accuracies2-accuracies1, marker='o', color='c', label=label2, linestyle='None', fillstyle='none', alpha=alpha)
    plt.scatter(tag_freqs, accuracies2-accuracies1, s=np.pi * (0.5 * (accuracies2-accuracies1)+10 )**2, c = np.random.rand(len(tag_freqs)), alpha=0.5)

    print 'annotating tags'
    texts = []
    for i, tag in enumerate(tags):
        #plt.annotate(tag, (tag_freqs[i], accuracies1[i]), xytext=(-10,10), \
        #        textcoords='offset points', ha='right', va='bottom', \
        #        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        #plt.annotate(tag, (tag_freqs[i], accuracies1[i]))
        #plt.annotate(tag, (tag_freqs[i], accuracies2[i]))
        plt.annotate(tag, (tag_freqs[i], accuracies2[i]-accuracies1[i]), horizontalalignment='center', verticalalignment='center', size=10+0.05*(accuracies2[i]-accuracies1[i]))
        #texts.append(plt.text(tag_freqs[i], accuracies1[i], tag))
    #adjust_text(texts, force_text=0.05, arrowprops=dict(arrowstyle="-|>", color='r', alpha=0.5))

    plt.xscale('symlog')
    #plt.yscale('log', basey=yscale_base)
    #plt.legend(loc='lower right', prop={'size':14})
    plt.xlabel('Frequency', size='large', fontweight='demibold')
    plt.ylabel('Increase in Accuracy', size='large', fontweight='demibold')
    #plt.ylim(ymax=1.05*scale_acc)
    plt.ylim(ymax=1.15*max(accuracies2-accuracies1))
    plt.xlim(min(tag_freqs) / 2, max(tag_freqs) * 5)
    plt.title(title, fontweight='demibold')
    plt.tight_layout()
    if filename:
        print 'saving plot to:', filename
        plt.savefig(filename)    


def get_flat_words(filename, encoding='utf-8'):
    """ Get flattened list of words """
    
    words = []
    with codecs.open(filename, encoding=encoding) as f:
        for line in f:
            for word in line.strip().split():
                words.append(word)
    return words
    

def get_ambig_words(filename, lbl_filename, encoding='utf-8'):
    """ Get ambiguous and non-ambiguous words """
    
    word2label = dict()
    with codecs.open(filename, encoding=encoding) as f:
        with codecs.open(lbl_filename, encoding=encoding) as f_lbl:
            for line, line_lbl in izip(f, f_lbl):
                for word, label in zip(line.strip().split(), line_lbl.strip().split()):
                    if word in word2label:
                        word2label[word].add(label)
                    else:
                        word2label[word] = {label}
    ambig, nonambig = set(), set()
    for word in word2label:
        if len(word2label[word]) > 1:
            ambig.add(word)
        elif len(word2label[word]) == 1:
            nonambig.add(word)
        else:
            sys.stderr.write('Warning: should not be here in get_ambig_words()\n')
    return ambig, nonambig                      
    

def eval_predictions_by_ambiguity(gold, pred, words, ambig, nonambig, tags, fig_pref=None, fig_num=1000):
    
    gold_ambig, pred_ambig, gold_nonambig, pred_nonambig, gold_unseen, pred_unseen = [], [], [], [], [], []
    for g, p, w in zip(gold, pred, words):
        if w in ambig:
            gold_ambig.append(g)
            pred_ambig.append(p)
        elif w in nonambig:
            gold_nonambig.append(g)
            pred_nonambig.append(p)
        else:
            gold_unseen.append(g)
            pred_unseen.append(p)
            
    print 'evaluating ambiguous words'
    fig_num = eval_predictions(gold_ambig, pred_ambig, tags, fig_pref + '.ambig.cm.png' if fig_pref else None, fig_num=fig_num)            
    print 'evaluating unambiguous words'
    fig_num = eval_predictions(gold_nonambig, pred_nonambig, tags, fig_pref + '.unambig.cm.png' if fig_pref else None, fig_num=fig_num)
    print 'evaluating unseen words'
    fig_num = eval_predictions(gold_unseen, pred_unseen, tags, fig_pref + '.unseen.cm.png' if fig_pref else None, fig_num=fig_num)
    return fig_num


def assign_majority_baseline(train_gold, train_words, test_words):
    """ Find the majority baseline
    
    Return tuple of (local_pred, global_pred) - the local and global majority baselines
    """

    tag2count, word2tag2count = dict(), dict()
    for g, w in zip(train_gold, train_words):
        tag2count[g] = tag2count.get(g, 0) + 1
        if w in word2tag2count:
            word2tag2count[w][g] = word2tag2count[w].get(g, 0) + 1
        else:
            word2tag2count[w] = {g: 1}
    majority_tag, majority_count = sorted(tag2count.items(), key=operator.itemgetter(1), reverse=True)[0]
    print 'global majority tag:', majority_tag, 'count:', majority_count
    local_pred = []
    for w in test_words:
        if w in word2tag2count:
            most_freq_tag, _ = sorted(word2tag2count[w].items(), key=operator.itemgetter(1), reverse=True)[0]
            local_pred.append(most_freq_tag)
        else:
            local_pred.append(majority_tag)
    global_pred = [majority_tag]*len(local_pred)
    return local_pred, global_pred


def run(gold_filename, pred_filename, train_lbl_filename=None, 
        train_filename=None, test_filename=None, fig_pref=None,
        test_pred_file2=None, label1='model 1', label2='model 2', annotation='', scale_acc=100.0):

    gold, pred, tags = get_predictions(gold_filename, pred_filename)    
    fig_num = 1
    fig_num = eval_predictions(gold, pred, tags, fig_pref + '.cm.png' if fig_pref else None, fig_num=fig_num)
    if test_pred_file2:
        gold2, pred2, tags2 = get_predictions(gold_filename, test_pred_file2)
    if train_lbl_filename:
        tag_freq_dict = get_freq_dict(train_lbl_filename)
        print 'evaluating predictions by tag frequency'
        print 'evaluating file:', pred_filename
        freqs, accuracies = eval_predictions_by_freq(gold, pred, tags, tag_freq_dict, scale_acc=scale_acc)
        plt.figure(fig_num)
        fig_num += 1
        fig_filename = fig_pref + '.tagfreq.png' if fig_pref else None
        fig_title = 'Accuracy per Tag Frequency' if annotation == '' else annotation + ' ' + 'Accuracy per Tag Frequency'
        plot_accuracy_by_freq(freqs, accuracies, filename=fig_filename, title=fig_title)
        if test_pred_file2:
            print 'evaluating file:', test_pred_file2
            freqs2, accuracies2 = eval_predictions_by_freq(gold2, pred2, tags2, tag_freq_dict, scale_acc=scale_acc)
            plt.figure(fig_num)
            fig_num += 1
            fig_filename = fig_pref + '.tagfreq.compare.png'
            plot_accuracy_by_freq_compare(freqs, accuracies, freqs2, accuracies2, label1, label2, fig_title, fig_filename, scale_acc=scale_acc)

            print 'evaluating by tag'
            tag_accs, sorted_tags = eval_predictions_by_tag(gold, pred, tags)
            tag_accs2, _ = eval_predictions_by_tag(gold2, pred2, tags2)
            plt.figure(fig_num, figsize=(9,6))
            fig_num += 1
            fig_filename = fig_pref + '.tag.compare.png'
            fig_title = 'Change in Accuracy per Tag Frequency' if annotation == '' else 'Change in ' + annotation + ' ' + 'Accuracy per Tag Frequency'
            plot_accuracy_by_tag_compare(tag_accs, tag_accs2, sorted_tags, tag_freq_dict, label1, label2, fig_title, fig_filename, scale_acc=scale_acc) 
        
        print 'evaluating predictions by cumulative minimal tag frequency'
        print 'evaluating file:', pred_filename
        freqs, accuracies = eval_predictions_by_freq(gold, pred, tags, tag_freq_dict, 
                                                     cumulative=True, scale_acc=scale_acc)
        plt.figure(fig_num)
        fig_num += 1
        fig_filename = fig_pref + '.tagfreq.cum.minfreq.png' if fig_pref else None
        fig_title = 'Accuracy per cumulative minimal Tag Frequency' if annotation == '' else annotation + ' ' + 'Accuracy per cumulative minimal Tag Frequency'
        plot_accuracy_by_freq(freqs, accuracies, filename=fig_filename, title=fig_title)
        if test_pred_file2:
            print 'evaluating file:', test_pred_file2
            freqs2, accuracies2 = eval_predictions_by_freq(gold2, pred2, tags2, 
                                                           tag_freq_dict, cumulative=True, scale_acc=scale_acc)
            plt.figure(fig_num)
            fig_num += 1
            fig_filename = fig_pref + '.tagfreq.cum.minfreq.compare.png'
            plot_accuracy_by_freq_compare(freqs, accuracies, freqs2, accuracies2, label1, label2, fig_title, fig_filename, scale_acc=scale_acc)
        
        print 'evaluating predictions by cumulative maximal tag frequency'
        print 'evaluating file:', pred_filename
        freqs, accuracies = eval_predictions_by_freq(gold, pred, tags, tag_freq_dict, 
                                                     cumulative=True, min_freq=False, scale_acc=scale_acc)
        plt.figure(fig_num)
        fig_num += 1
        fig_filename = fig_pref + '.tagfreq.cum.maxfreq.png' if fig_pref else None
        fig_title = 'Accuracy per cumulative maximal Tag Frequency' if annotation == '' else annotation + ' ' + 'Accuracy per cumulative maximal Tag Frequency'
        plot_accuracy_by_freq(freqs, accuracies, filename=fig_filename, title=fig_title)
        if test_pred_file2:
            print 'evaluating file:', test_pred_file2
            freqs2, accuracies2 = eval_predictions_by_freq(gold2, pred2, tags2, 
                                                           tag_freq_dict, cumulative=True, min_freq=False, scale_acc=scale_acc)
            plt.figure(fig_num)
            fig_num += 1
            fig_filename = fig_pref + '.tagfreq.cum.maxfreq.compare.png'
            plot_accuracy_by_freq_compare(freqs, accuracies, freqs2, accuracies2, label1, label2, fig_title, fig_filename, scale_acc=scale_acc)


    if train_lbl_filename and train_filename and test_filename:
        # majority baseline
        print 'getting majority baseline'
        train_gold = get_flat_words(train_lbl_filename)
        train_words = get_flat_words(train_filename)
        test_words = get_flat_words(test_filename)
        pred_majority_local, pred_majority_global = assign_majority_baseline(train_gold, train_words, test_words)
        print 'evaluating global majority baseline'
        eval_predictions(gold, pred_majority_global, tags)
        print 'evaluating local majority baseline'
        eval_predictions(gold, pred_majority_local, tags)       


    if train_filename and test_filename:
        word_freq_dict = get_freq_dict(train_filename)
        words = get_flat_words(test_filename)
        num_oov = len([word for word in words if word not in word_freq_dict])
        print 'number of oov words in test file: {}, fraction: {}'.format(num_oov, np.round(100.0*num_oov/len(words), 2))
        print 'evaluating predictions by word frequency'
        print 'evaluating file:', pred_filename
        freqs, accuracies = eval_predictions_by_freq(gold, pred, tags, word_freq_dict, 
                                                     words=words, scale_acc=scale_acc)
        plt.figure(fig_num)
        fig_num += 1
        fig_filename = fig_pref + '.wordfreq.png' if fig_pref else None
        fig_title = 'Accuracy per Word Frequency' if annotation == '' else annotation + ' ' + 'Accuracy per Word Frequency'
        plot_accuracy_by_freq(freqs, accuracies, filename=fig_filename, title=fig_title)
        if test_pred_file2:
            print 'evaluating file:', test_pred_file2
            freqs2, accuracies2 = eval_predictions_by_freq(gold2, pred2, tags2, word_freq_dict, words=words, scale_acc=scale_acc)

            plt.figure(fig_num)
            fig_num += 1
            fig_filename = fig_pref + '.wordfreq.histogram.compare.png'
            plot_accuracy_by_freq_compare_histogram(freqs, accuracies, freqs2, accuracies2, label1, label2, fig_title, fig_filename, scale_acc=scale_acc)

            plt.figure(fig_num)
            fig_num += 1
            fig_filename = fig_pref + '.wordfreq.compare.png'
            plot_accuracy_by_freq_compare(freqs, accuracies, freqs2, accuracies2, label1, label2, fig_title, fig_filename, scale_acc=scale_acc)
        
        print 'evaluating predictions by cumulative minimal word frequendcy'
        print 'evaluating file:', pred_filename
        freqs, accuracies = eval_predictions_by_freq(gold, pred, tags, word_freq_dict, 
                                                     words=words, cumulative=True, scale_acc=scale_acc)
        plt.figure(fig_num)
        fig_num += 1
        fig_filename = fig_pref + '.wordfreq.cum.minfreq.png' if fig_pref else None
        fig_title = 'Accuracy per cumulative minimal Word Frequency' if annotation == '' else annotation + ' ' + 'Accuracy per cumulative minimal Word Frequency'
        plot_accuracy_by_freq(freqs, accuracies, filename=fig_filename, title=fig_title)
        if test_pred_file2:
            print 'evaluating file:', test_pred_file2
            freqs2, accuracies2 = eval_predictions_by_freq(gold2, pred2, tags2, 
                                                           word_freq_dict, words=words, cumulative=True, scale_acc=scale_acc)
            plt.figure(fig_num)
            fig_num += 1
            fig_filename = fig_pref + '.wordfreq.cum.minfreq.compare.png' 
            plot_accuracy_by_freq_compare(freqs, accuracies, freqs2, accuracies2, label1, label2, fig_title, fig_filename, scale_acc=scale_acc)
            
        print 'evaluating predictions by cumulative maximal word frequendcy'
        print 'evaluating file:', pred_filename
        freqs, accuracies = eval_predictions_by_freq(gold, pred, tags, word_freq_dict, 
                                                     words=words, cumulative=True, min_freq=False, scale_acc=scale_acc)
        plt.figure(fig_num)
        fig_num += 1
        fig_filename = fig_pref + '.wordfreq.cum.maxfreq.png' if fig_pref else None
        fig_title = 'Accuracy per cumulative maximal Word Frequency' if annotation == '' else annotation + ' ' + 'Accuracy per cumulative maximal Word Frequency'
        plot_accuracy_by_freq(freqs, accuracies, filename=fig_filename, title=fig_title)
        if test_pred_file2:
            print 'evaluating file:', test_pred_file2
            freqs2, accuracies2 = eval_predictions_by_freq(gold2, pred2, tags2, 
                                                           word_freq_dict, words=words, cumulative=True, min_freq=False, scale_acc=scale_acc)
            plt.figure(fig_num)
            fig_num += 1
            fig_filename = fig_pref + '.wordfreq.cum.maxfreq.compare.png'
            plot_accuracy_by_freq_compare(freqs, accuracies, freqs2, accuracies2, label1, label2, fig_title, fig_filename, scale_acc=scale_acc)
    
        ## eval by ambiguity
        if train_filename and train_lbl_filename:
            ambig, nonambig = get_ambig_words(train_filename, train_lbl_filename)
            fig_num = eval_predictions_by_ambiguity(gold, pred, words, ambig, nonambig, tags, fig_pref, fig_num)
        
            if test_pred_file2:
                fig_num = eval_predictions_by_ambiguity(gold2, pred2, words, ambig, nonambig, tags2, fig_pref + '.compare', fig_num)
    
    """
    # TODO remove unseen tags for confusion matrix labels (need to shift tag indices)
    # eval by test tag freq
    min_freq = 100
    print 'filtering by minimum test tag frequency:', min_freq
    test_tag_freq_dict = get_freq_dict(gold_filename)
    filtered_gold, filtered_pred = filter_by_min_tag_freq(gold, pred, min_freq, test_tag_freq_dict)
    fig_num = eval_predictions(filtered_gold, filtered_pred, tags, fig_pref + '.test.tagfreq.min.cm.png' if fig_pref else None, fig_num=fig_num)
    """
    



if __name__ == '__main__':
    args = docopt(__doc__)
    
    if args['--annotation'] and args['--annotation'].lower() == 'pos':
        annotation = 'POS'
    elif args['--annotation'] and args['--annotation'].lower() in ['morph', 'morphology']:
        annotation = 'Morphology'
    elif args['--annotation'] and args['--annotation'].lower() in ['stag', 'semtag', 'semtags']:
        annotation = 'Semtag'
    else:
        annotation = ''
        
    run(args['TEST_GOLD_FILE'], args['TEST_PRED_FILE'], 
        args['--train_lbl_file'], args['--train_file'], args['--test_file'], 
        args['--fig_pref'], args['--test_pred_file2'], args['--label1'], args['--label2'], annotation=annotation)
    
