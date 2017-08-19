# -*- coding: utf-8 -*-
"""
Extract dependencies from a Semantic Dependency Parsing treebank.

Usage: extract_sdp.py [--sep sep] [--first_arg_col first_arg_col] IN_FILE OUT_TEXT_FILE OUT_HEAD_FILE OUT_DEPREL_FILE

Arguments:
  IN_FILE            SDP file in sdp format
  OUT_TEXT_FILE      File to write raw texts, one sentence per line
  OUT_HEAD_FILE      File to write heads, which are either an ID (1-indexed) or 0 (for no dependency)
                     If a word has more than one head, then its heads will be sparated by --sep
  OUT_DEPREL_FILE    File to write UD relations to the head
                     IF a word has more than one head, then its relations to the heads will be separated by --sep

Options:
  -h, --help                       show this help message  
  --sep sep                        separator for multiple heads (Default: "|")
  --first_arg_col first_arg_col    first argument column id (0-indexed) (Default: 7)
  
"""

from docopt import docopt
import codecs


def run(sdp_file, out_text_file, out_head_file, out_deprel_file, sep, first_arg_col, encoding='UTF-8'):
    
    with codecs.open(sdp_file, encoding=encoding) as f_sdp:
        with codecs.open(out_text_file, 'w', encoding=encoding) as f_out_text:
            with codecs.open(out_head_file, 'w', encoding=encoding) as f_out_head:
                with codecs.open(out_deprel_file, 'w', encoding=encoding) as f_out_deprel:
                    words, rels, preds, pred_ids = [], [], [], []
                    tok_id = 0
                    for line in f_sdp:
                        #print line
                        if line.startswith('#'):
                            continue
                        if line.strip() == '':
                            # map pred order to id, then join multiple heads
                            heads = []
                            for cur_preds in preds:
                                if len(cur_preds) > 0:
                                    heads.append(sep.join([str(pred_ids[cur_pred]) for cur_pred in cur_preds]))
                                else:
                                    heads.append('0')
                                    
                            if len(words) > 0:
                                f_out_text.write(' '.join(words) + '\n')
                                f_out_deprel.write(' '.join(rels) + '\n')                                                                
                                f_out_head.write(' '.join(heads) + '\n')

                            words, rels, preds, pred_ids = [], [], [], []
                            tok_id = 0
                            continue
                        splt = line.strip().split('\t')
                        tok_id += 1
                        
                        # is predicate
                        if splt[5] == '+':
                            pred_ids.append(tok_id)
                        
                        words.append(splt[1])
                        cur_preds, cur_rels = [], []
                        # look for arguments
                        for i in xrange(first_arg_col, len(splt)):
                            # is argument
                            if splt[i] != '_':
                                # get the pred's order
                                cur_preds.append(i-first_arg_col)
                                cur_rels.append(splt[i])
                        preds.append(cur_preds)
                        if len(cur_rels) > 0:
                            rels.append(sep.join(cur_rels))
                        else:
                            rels.append('_')
                            
                        
                        



if __name__ == '__main__':
    args = docopt(__doc__)
    sep = '|'
    if args['--sep']:
        sep = args['--sep']
    first_arg_col = 7
    if args['--first_arg_col']:
        first_arg_col = args['--first_arg_col']
    
    run(args['IN_FILE'], args['OUT_TEXT_FILE'], args['OUT_HEAD_FILE'], args['OUT_DEPREL_FILE'], sep=sep, first_arg_col=first_arg_col)

