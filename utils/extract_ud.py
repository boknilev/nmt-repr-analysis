# -*- coding: utf-8 -*-
"""
Extract dependencies from a Universal Dependencies treebank.

Usage: extract_ud.py IN_FILE OUT_TEXT_FILE OUT_HEAD_FILE OUT_DEPREL_FILE

Arguments:
  IN_FILE            UD file in conllu format
  OUT_TEXT_FILE      File to write raw texts, one sentence per line
  OUT_HEAD_FILE      File to write heads, which are either an ID (1-indexed) or 0 (for root)
  OUT_DEPREL_FILE    File to write UD relations to the head

Options:
  -h, --help                     show this help message  

"""

from docopt import docopt
import codecs


def run(ud_file, out_text_file, out_head_file, out_deprel_file, encoding='UTF-8'):
    
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
                        words.append(splt[1])
                        heads.append(splt[6])
                        rels.append(splt[7])
                        
                        



if __name__ == '__main__':
    args = docopt(__doc__)
    
    run(args['IN_FILE'], args['OUT_TEXT_FILE'], args['OUT_HEAD_FILE'], args['OUT_DEPREL_FILE'])

