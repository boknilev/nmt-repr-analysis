# -*- coding: utf-8 -*-
"""
Extract texts with annotations from semtags file

Usage: extract_semtags.py [--split_mwe] IN_FILE OUT_FILE OUT_LABEL_FILE

Arguments:
  IN_FILE           semtags .conll file
  OUT_FILE          File to write raw texts, one sentence per line
  OUT_LABEL_FILE    File to write corresponding labels, one sentence per line
  
Options:
  -h, --help                     show this help message  
  --split_mew                    Split multi-word expressions 
"""


from docopt import docopt


def convert_file(in_file, out_file, out_label_file, split_mwe=False):
    
    with open(in_file) as f_in:
        with open(out_file, 'w') as f_out:
            with open(out_label_file, 'w') as f_out_label:
                first_word = True
                tags = set()
                for line in f_in:
                    if line.strip() == '':
                        f_out.write('\n')
                        f_out_label.write('\n')
                        first_word = True
                        continue
                    if first_word:
                        first_word = False
                    else:
                        f_out.write(' ')
                        f_out_label.write(' ')
                    word, tag = line.strip().split()
                    tags.add(tag)
                    # handle multi-word expressions
                    if split_mwe and '~' in word:
                        words = word.split('~')
                        f_out.write(' '.join(words))
                        f_out_label.write(' '.join([tag]*len(words)))
                    else:
                        f_out.write(word)
                        f_out_label.write(tag)
    print 'written sentences to:', out_file
    print 'written labels to:', out_label_file                     
    print 'found', len(tags), 'unique tags'



if __name__ == '__main__':
    args = docopt(__doc__)
    
    split_mwe = False
    if args['--split_mwe']:
        split_mwe = True 
        print 'splitting MWEs'
    
    convert_file(args['IN_FILE'], args['OUT_FILE'], args['OUT_LABEL_FILE'], split_mwe) 

