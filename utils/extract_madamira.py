# -*- coding: utf-8 -*-
"""
Extract texts with annotations from MADAMIRA output file

Usage: extract_madamira.py [--annotation annotation] [--feature_map_file map_file] IN_FILE OUT_FILE OUT_LABEL_FILE

Arguments:
  IN_FILE           MADAMIRA .mada file
  OUT_FILE          File to write raw texts, one sentence per line
  OUT_LABEL_FILE    File to write corresponding labels, one sentence per line
  
Options:
  -h, --help                     show this help message  
  --annotation annotation        Annotation to extract (POS, morph, etc.)
  --feature_map_file map_file    File containing mapping between features (to convert SAMA to PATB tags)
"""


from docopt import docopt
import os, sys, codecs
import extract_patb as patb_extractor

LATIN_PREFIX = '@@LAT@@'
NO_ANALYSIS = 'NO-ANALYSIS'
LATIN_ANALYSIS = 'LATIN-ANALYSIS'



class WordAnalysis(object):
    
    def __init__(self, word, bw, no_analysis):
        
        if word.startswith(LATIN_PREFIX):
            self.is_latin = True
            self.word = word[len(LATIN_PREFIX):]
        else:
            self.is_latin = False
            self.word = word
        self.bw = bw
        self.no_analysis = no_analysis
    
    def get_label(self, annotation, feat_map):
        if self.is_latin:
            return LATIN_ANALYSIS
        if self.no_analysis or self.bw == '':
            return NO_ANALYSIS          
        # handle special cases 
        if self.word == '+':
            pos = self.bw.split('/')[1]
            return patb_extractor.get_label(pos, annotation, feat_map)
        if self.word == '/':
            pos = self.bw.split('/')[-1]
            return patb_extractor.get_label(pos, annotation, feat_map)
        # in case there's more than one '+' or more than one '/'
        if '+' in self.word:
            sys.stderr.write('Warning: word ' + self.word + ' has \'+\' sign, assuming Latin analysis\n')
            return LATIN_ANALYSIS
        if '/' in self.word:
            sys.stderr.write('Warning: word ' + self.word + ' has \'/\' sign, assuming Latin analysis\n')
            return LATIN_ANALYSIS

        bw_parts = self.bw.split('+')
        tags = [part.split('/')[1] for part in bw_parts]
        pos = '+'.join(tags)
        return patb_extractor.get_label(pos, annotation, feat_map)


def process_file(mada_file, out_file, out_label_file, annotation, feat_map):
    
    print 'processing file:', mada_file
    with codecs.open(mada_file, encoding='utf-8') as f_mada:
        with codecs.open(out_file, 'w', encoding='utf-8') as f_out:
            with open(out_label_file, 'w') as f_out_label:
                word, bw, no_analysis  = '', '', None
                words, labels = [], []
                counter = 0
                for line in f_mada:
                    if line.strip() == 'SENTENCE BREAK':
                        counter += 1
                        if counter % 10000 == 0:
                            print 'sentence:', counter
                        if len(words) != len(labels):
                            sys.stderr.write('Warning: different numbers of words and labels after line ' + str(counter) + '\n')
                        f_out.write(' '.join(words) + '\n')
                        f_out_label.write(' '.join(labels) + '\n')
                        word, bw, no_analysis  = '', '', None                        
                        words, labels = [], []
                    elif line.strip().startswith(';;WORD'):
                        word = line.strip().split()[1]
                    elif line.strip() == ';;NO-ANALYSIS':
                        no_analysis = True
                    elif line.strip().startswith('*'):
                        no_analysis = False
                        for pair in line.strip().split()[1:]:
                            feat, val = pair.split(':', 1)
                            if feat == 'bw':
                                bw = val
                                break
                    elif line.strip() == '--------------':
                        if word != '':
                            word_analysis = WordAnalysis(word, bw, no_analysis)
                            words.append(word_analysis.word)
                            label = word_analysis.get_label(annotation, feat_map)
                            labels.append(label)
    print 'written words to:', out_file
    print 'written labels to:', out_label_file
                
    
            
def run(mada_file, out_file, out_label_file, annotation, feat_map_file):

    feat_map = None
    if feat_map_file and os.path.isfile(feat_map_file):
        feat_map = patb_extractor.load_feature_map(feat_map_file)
    process_file(mada_file, out_file, out_label_file, annotation, feat_map)    
        
    
    
if __name__ == '__main__':
    args = docopt(__doc__)
    
    if args['--annotation'] and args['--annotation'] in ['pos', 'morph']:
        annotation = args['--annotation']
    else:
        sys.stderr.write('Unknown or missing annotation: ' + str(args['--annotation']) + ', using default morph\n')
        annotation = 'morph'
    
    run(args['IN_FILE'], args['OUT_FILE'], args['OUT_LABEL_FILE'], annotation, args['--feature_map_file']) 
    
    
    
    
    
    
