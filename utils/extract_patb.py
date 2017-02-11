# -*- coding: utf-8 -*-
"""
Extract texts with annotations from the Penn Arabic Treebank

Usage: extract_patb.py [--annotation annotation] [--feature_map_file map_file] IN_FILE_LIST IN_DIR OUT_FILE OUT_LABEL_FILE

Arguments:
  IN_FILE_LIST      List of PATB pos files (before treebanking)
  IN_DIR            Directory containing PATB files
  OUT_FILE          File to write raw texts, one sentence per line
  OUT_LABEL_FILE    File to write corresponding labels, one sentence per line
  
Options:
  -h, --help                     show this help message  
  --annotation annotation        Annotation to extract (POS, morph, etc.)
  --feature_map_file map_file    File containing mapping between features (to convert SAMA to PATB tags)
"""


from docopt import docopt
import os, sys, codecs

PREFIXES = {'CONJ', 'PREP', 'SUB_CONJ', 'CONNEC_PART', 'EMPHATIC_PART', 
            'FOCUS_PART', 'FUT_PART', 'INTERROG_PART', 'JUS_PART', 'NEG_PART', 
            'PART', 'RC_PART', 'RESTRIC_PART', 'VERB_PART', 'VOC_PART'}
SUFFIXES = {'POSS_PRON_1P', 'POSS_PRON_1S', 'POSS_PRON_2D', 'POSS_PRON_2FP', 
            'POSS_PRON_2FS', 'POSS_PRON_2MP', 'POSS_PRON_2MS', 'POSS_PRON_3D', 
            'POSS_PRON_3FP', 'POSS_PRON_3FS', 'POSS_PRON_3MP', 'POSS_PRON_3MS',
            'CVSUFF_DO:1P', 'CVSUFF_DO:1S', 'CVSUFF_DO:2D', 'CVSUFF_DO:2FP', 
            'CVSUFF_DO:2FS', 'CVSUFF_DO:2MP', 'CVSUFF_DO:2MS', 'CVSUFF_DO:3D', 
            'CVSUFF_DO:3FP', 'CVSUFF_DO:3FS', 'CVSUFF_DO:3MP', 'CVSUFF_DO:3MS',
            'IVSUFF_DO:1P', 'IVSUFF_DO:1S', 'IVSUFF_DO:2D', 'IVSUFF_DO:2FP', 
            'IVSUFF_DO:2FS', 'IVSUFF_DO:2MP', 'IVSUFF_DO:2MS', 'IVSUFF_DO:3D', 
            'IVSUFF_DO:3FP', 'IVSUFF_DO:3FS', 'IVSUFF_DO:3MP', 'IVSUFF_DO:3MS',
            'PVSUFF_DO:1P', 'PVSUFF_DO:1S', 'PVSUFF_DO:2D', 'PVSUFF_DO:2FP', 
            'PVSUFF_DO:2FS', 'PVSUFF_DO:2MP', 'PVSUFF_DO:2MS', 'PVSUFF_DO:3D', 
            'PVSUFF_DO:3FP', 'PVSUFF_DO:3FS', 'PVSUFF_DO:3MP', 'PVSUFF_DO:3MS',
            'PRON_1P', 'PRON_1S', 'PRON_2D', 'PRON_2FP', 'PRON_2FS', 
            'PRON_2MP', 'PRON_2MS', 'PRON_3D', 'PRON_3FP', 'PRON_3FS', 
            'PRON_3MP', 'PRON_3MS'}
            
# exception rules for feature map (needed because working with "before" pos file)
FEAT_MAP_EXCEPTIONS = {'NOUN+CASE_DEF_ACC+SUB_CONJ':'IN', 'NOUN+CASE_DEF_ACC+REL_PRON':'WP', 'PSEUDO_VERB+REL_PRON':'WP'}
# fall back to some basic categories, in order of more to less specific roughly
FALLBACK_TAGS = ['DET+NOUN_PROP', 'DET+NOUN', 'NOUN_PROP', 'NOUN', 'IV', 'PV', 'VERB', 'ABBREV', 'DET+ADJ_COMP', 'DET+ADJ_NUM', 'DET+ADJ', 'ADJ_COMP', 'ADJ_NUM', 'ADJ', 'INTERROG_PRON']

def load_feature_map(feature_map_file):
    """ Load mapping between features (converting SAMA to PATB tags) 
    
    feature_map_file: file containing mapping provided with PATB, e.g. atb1-v4.1-taglist-conversion-to-PennPOS-forrelease.lisp    
    """
    
    print 'loading feature map from:', feature_map_file
    feature_map = dict()
    with open(feature_map_file) as f:
        for line in f:
            # example line: "    (NOUN.VN+CASE_DEF_ACC VBG)"
            if line.startswith('    (') and line.rstrip().endswith(')'):
                morph, pos = line.strip()[1:-1].split(' ')
                if morph in feature_map:
                    sys.stderr.write('Warning: already saw feature ' + morph + ' with pos ' + feature_map[morph] + ', now with pos ' + pos + '\n')
                else:
                    feature_map[morph] = pos
    # add some hard-coded exceptional cases
    for k in FEAT_MAP_EXCEPTIONS:
        if k not in feature_map:
            feature_map[k] = FEAT_MAP_EXCEPTIONS[k]
    return feature_map    
    

def get_label(pos, annotation, feat_map=None):
    
    # if wanted annotation is POS and we know the conversion
    if annotation == 'pos' and feat_map:
        if pos in feat_map:
            return feat_map[pos]
        # try some fallbacks
        else:
            #sys.stderr.write('Warning: pos ' + pos + ' not found in feat_map, attempting to fall back by ignoring certain prefixes/suffixes\n')
            # the feature map list contains tags after treebank tokenization, so we might miss some cases
            # e.g. PREP+PRON_3FS --> , CONJ+DET+NOUN+CASE_DEF_GEN -->
            # first try to remove suffixes
            new_tags = pos.split('+')
            new_pos = pos
            if new_tags[-1] in SUFFIXES:
                new_tags = new_tags[:-1]
                new_pos = '+'.join(new_tags)
                if new_pos in feat_map:
                    return feat_map[new_pos]
            # next try to remove prefixes, allowing up to two
            if new_tags[0] in PREFIXES:
                new_tags = new_tags[1:]
                new_pos = '+'.join(new_tags)
                if new_pos in feat_map:
                    return feat_map[new_pos]
                if new_tags[0] in PREFIXES:
                    new_tags = new_tags[1:]
                    new_pos = '+'.join(new_tags)
                    if new_pos in feat_map:
                        return feat_map[new_pos]
            # fall back to some basic categories, if their tags exist
            for tag in FALLBACK_TAGS:
                if tag in new_pos and tag in feat_map:
                    sys.stderr.write('Warning: falling back to basic tag ' + tag + ' for pos ' + new_pos + '\n')
                    return feat_map[tag]
            sys.stderr.write('Warning: could not fall back, will return original pos: ' + pos + '\n')
            
    # else return the morph features
    return pos


def process_file(patb_file, annotation, feat_map=None):
    """ Extract annotations from a PATB file
    
    patb_file: a file containing PATB annotations, from pos/before folder
    annotation: desired annotation, can be pos (part-of-speech) or morph (morphological features)
    feat_map: optional dictionary mapping feature tags to other tags (to convert SAMA to PATB POS tags)
    
    return (words, labels): tuple of two lists:
                            words: list of lists of raw words
                            labels: list of lists of labels 
    """
    
    with codecs.open(patb_file, encoding='utf-8') as f:
        words, labels, sent_words, sent_labels = [], [], [], []
        input_string, index, pos = '', '', ''
        sent_num = 1
        for line in f:
            if line.strip() == '':
                # if started new sentence 
                if int(index[1:].split('W')[0]) == sent_num + 1:
                    words.append(sent_words)
                    labels.append(sent_labels)
                    sent_words = [input_string]
                    sent_labels = [get_label(pos, annotation, feat_map)]
                    sent_num += 1
                # continue current sentence
                else:
                    sent_words.append(input_string)
                    sent_labels.append(get_label(pos, annotation, feat_map))
            else:
                field, value = line.strip().split(':', 1)
                field = field.strip(); value = value.strip()
                if field == 'INPUT STRING':
                    input_string = value
                elif field == 'INDEX':
                    index = value
                elif field == 'POS':
                    pos = value  
    # add last sentence
    if len(words) != 0:
        words.append(sent_words)
        labels.append(sent_labels)
                   
    #print words
    return words, labels
    

def process_file_list(patb_file_list, patb_dir, out_file, out_label_file, annotation, feat_map=None):

    with codecs.open(out_file, 'w', encoding='utf-8') as f_out:
        with codecs.open(out_label_file, 'w', encoding='utf-8') as f_out_label:
            for patb_file in patb_file_list:
                #print 'processing file:', patb_file
                patb_file_path = os.path.join(patb_dir, patb_file)
                if os.path.isfile(patb_file_path):
                    words, labels = process_file(patb_file_path, annotation, feat_map)
                    for sent_words, sent_labels in zip(words, labels):
                        f_out.write(' '.join(sent_words) + '\n')
                        f_out_label.write(' '.join(sent_labels) + '\n')
    print 'written texts to:', out_file
    print 'written labels to:', out_label_file


def run(patb_file_list_file, patb_dir, out_file, out_label_file, annotation, feat_map_file=None):
    
    with open(patb_file_list_file) as f:
        patb_file_list = [line.strip() for line in f.readlines()]
    feat_map = None
    if feat_map_file and os.path.isfile(feat_map_file):
        feat_map = load_feature_map(feat_map_file)
    process_file_list(patb_file_list, patb_dir, out_file, out_label_file, annotation, feat_map)    
    

if __name__ == '__main__':
    args = docopt(__doc__)
    
    if args['--annotation'] and args['--annotation'] in ['pos', 'morph']:
        annotation = args['--annotation']
    else:
        sys.stderr.write('Unknown or missing annotation: ' + str(args['--annotation']) + ', using default morph\n')
        annotation = 'morph'
    
    run(args['IN_FILE_LIST'], args['IN_DIR'], args['OUT_FILE'], args['OUT_LABEL_FILE'], annotation, args['--feature_map_file']) 
    
    
    
    
    
    