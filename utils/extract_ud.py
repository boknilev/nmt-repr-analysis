# -*- coding: utf-8 -*-
"""
Extract dependencies from a Universal Dependencies treebank.

Usage: extract_ud.py [--merge_toks] IN_FILE OUT_TEXT_FILE OUT_HEAD_FILE OUT_DEPREL_FILE

Arguments:
  IN_FILE            UD file in conllu format
  OUT_TEXT_FILE      File to write raw texts, one sentence per line
  OUT_HEAD_FILE      File to write heads, which are either an ID (1-indexed) or 0 (for root)
  OUT_DEPREL_FILE    File to write UD relations to the head

Options:
  -h, --help                     show this help message  
  --merge_toks                   merge tokens that are spearated by space (happens with numbers in the French treebank, e.g. 10 000)

"""

from docopt import docopt
import codecs


def run(ud_file, out_text_file, out_head_file, out_deprel_file, merge_toks=False, encoding='UTF-8'):
    
    with codecs.open(ud_file, encoding=encoding) as f_ud:
        with codecs.open(out_text_file, 'w', encoding=encoding) as f_out_text:
            with codecs.open(out_head_file, 'w', encoding=encoding) as f_out_head:
                with codecs.open(out_deprel_file, 'w', encoding=encoding) as f_out_deprel:
                    words, heads, rels = [], [], []
                    for line in f_ud:
                        if line.startswith('#'):
                            continue
                        if line.strip() == '':
                            f_out_text.write(' '.join(words) + '\n')
                            f_out_head.write(' '.join(heads) + '\n')
                            f_out_deprel.write(' '.join(rels) + '\n')
                            words, heads, rels = [], [], []
                            continue
                        splt = line.strip().split('\t')
                        # skip multiword tokens and empty nodes
                        # TODO consider this
                        if '-' in splt[0] or '.' in splt[0]:
                            continue
                        if merge_toks:
                            word = ''.join(splt[1].split())
                        else:
                            word = splt[1]
                        words.append(word)
                        heads.append(splt[6])
                        rels.append(splt[7])
                        
                        



if __name__ == '__main__':
    args = docopt(__doc__)
    merge_toks = False
    if args['--merge_toks']:
        merge_toks = True
    
    run(args['IN_FILE'], args['OUT_TEXT_FILE'], args['OUT_HEAD_FILE'], args['OUT_DEPREL_FILE'], merge_toks=merge_toks)

