import argparse
import csv
import pdb
import itertools
import cPickle as pickle

from stanfordcorenlp import StanfordCoreNLP

def get_orig_fnplus():
  #with open("../entailment_data/fnplus_orig/FN+/FNPlus-release/fnplus-fulltext", "rb") as csvfile:
  with open("../entailment_data/fnplus_orig/FN+/all-data/fulltext", "rb") as csvfile:
    csvreader = csv.reader(csvfile, delimiter='\t')
    sent2paraphrases = {}
    for row in csvreader:
      sent = row[-1].replace("[[", "").replace("]]", "")
      if sent not in sent2paraphrases:
        sent2paraphrases[sent] = set()
      orig_word = sent[-2]
      paraphrase = sent[-3]
      sent2paraphrases[sent].add((orig_word, paraphrase))

  return sent2paraphrases

def get_source(input_f):
  instances = {}
  sent_count = 0
  for line in open(input_f):
    line = line.split("|||")
    context = line[0]
    hyp = line[1]
    instances[sent_count] = (context, hyp)
    sent_count += 1
  return instances

def get_diff_loc(s1, s2):
  s1 = s1.split()
  s2 = s2.split()

  assert len(s1) == len(s2), "Context & hypothesis lengths are difference"
  for _ in range(len(s1)):
    if s1[_] != s2[_]:
      return _

  return -1


def split_data_by_grammar(instances, corenlp_path):
  nlp = StanfordCoreNLP(corenlp_path)
  same_sents_count = 0
  example_loc_to_grammar = {} # dictionary from number to 0 (indicating diff pos tag), 1 (indicating same pos tag)

  tags = set()

  for key,pair in instances.items():
    #if key > 20:
    #  nlp.close()
    #  return example_loc_to_grammar
    if key % 250 == 0:
      print key
    context = pair[0].strip()
    hyp = pair[1].strip()

    diff_loc = get_diff_loc(context, hyp)
    if diff_loc == -1:
      same_sents_count += 1
      example_loc_to_grammar[key] = 1 
      continue

    context_pos = nlp.pos_tag(context)
    hyp_pos = nlp.pos_tag(hyp)
    tags.add(hyp_pos[diff_loc][-1])
    tags.add(context_pos[diff_loc][-1])

    #print context_pos[diff_loc][-1][0],  hyp_pos[diff_loc][-1][0]
    if context_pos[diff_loc][-1] ==  hyp_pos[diff_loc][-1]:
      example_loc_to_grammar[key] = 1 #if context_pos[diff_loc][-1] ==  hyp_pos[diff_loc][-1] else 0
    else:
      example_loc_to_grammar[key] = 0

  nlp.close()

  print "Same sents count: %d" % (same_sents_count)
  pdb.set_trace()
  return example_loc_to_grammar

def compute_accuracies(example_loc, gold_f, ar_f, de_f, zh_f, es_f):
  line_count = 0
  pos_same_2_count = {1: [0, 0, 0, 0, 0], 0: [0, 0, 0, 0, 0]}
  for gold, ar, de, es, zh in itertools.izip(open(gold_f), open(ar_f), open(de_f), open(es_f), open(zh_f)):
  #for gold, ar, de, zh, es in itertools.izip(open(gold_f), open(ar_f), open(de_f), open(zh_f), open(es_f)):
    pos_same = example_loc[line_count]
    line_count += 1
    #pdb.set_trace()

    pos_same_2_count[pos_same][0] += 1
    if gold.strip() == ar.strip():
      pos_same_2_count[pos_same][1] += 1
    if gold.strip() == de.strip():
      pos_same_2_count[pos_same][4] += 1
    if gold.strip() == es.strip():
      pos_same_2_count[pos_same][2] += 1
    if gold.strip() == zh.strip():
      pos_same_2_count[pos_same][3] += 1

  for key in pos_same_2_count:
    for i in range(1, len(pos_same_2_count[key])):
      pos_same_2_count[key][i] = float(pos_same_2_count[key][i]) / pos_same_2_count[key][0]

  return pos_same_2_count

def print_latex_table(accuracies):
  table= "\\begin{tabular}{l|cccc|c} \n\
    \\toprule %\hline \n\
    	 & ar & es & zh & de & MAJ \\\\ \\midrule %\hline \n "
  tot = float(accuracies[1][0] + accuracies[0][0])
  table += "Same Tag & " + " & ".join(["%.2f" % (100*item) for item in accuracies[1][1:]]) + " & %.2f \\\\ \n" % (100*accuracies[1][0] / tot)
  table += "Different Tag & " + " & ".join(["%.2f" % (100*item) for item in accuracies[0][1:]]) + " & %.2f \\\\ \n" % (100*accuracies[0][0] / tot)
  table += "\\bottomrule %\hline \n \
    \\end{tabular}"
  print table

def main(args):
  instances = get_source(args.src)
  print "got instances"

  example_loc_to_grammar = ""
  if args.pos_tags:
    example_loc_to_grammar = pickle.load(open(args.pos_tags, "rb"))
  else:
    example_loc_to_grammar = split_data_by_grammar(instances, args.corenlp_path)
    pickle.dump(example_loc_to_grammar, open("example_loc_to_grammar.pkl", "wb"))

  accuracies = compute_accuracies(example_loc_to_grammar, args.gold, args.ar_pred, args.de_pred, args.zh_pred, args.es_pred) 

  print_latex_table(accuracies)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Analysis based on POS tags in FN+.')
  parser.add_argument('--src', help="path to source file")
  parser.add_argument('--gold', help="path to gold labels file")
  parser.add_argument('--ar_pred', help="path to pred labels file")
  parser.add_argument('--de_pred', help="path to pred labels file")
  parser.add_argument('--zh_pred', help="path to pred labels file")
  parser.add_argument('--es_pred', help="path to pred labels file")
  parser.add_argument('--corenlp_path', help="path to CoreNLP directory")
  parser.add_argument('--pos_tags', help="Path to pickle file of src with POS tags")

  args = parser.parse_args()

  main(args)


'''
 pred_ar=open("/export/ssd/apoliak/nmt-repr-anaysis-sem/output/fnplus-fnplus/linear_classifier/en-ar/val_pred_file.epoch1")
 pred_es=open("/export/ssd/apoliak/nmt-repr-anaysis-sem/output/fnplus-fnplus/linear_classifier/en-es/val_pred_file.epoch1")
 pred_zh=open("/export/ssd/apoliak/nmt-repr-anaysis-sem/output/fnplus-fnplus/linear_classifier/en-zh/test_pred_file.epoch1")
 pred_de=open("/export/ssd/apoliak/nmt-repr-anaysis-sem/output/fnplus-fnplus/linear_classifier/en-de/val_pred_file.epoch1")

 gold=open("/export/ssd/apoliak/nmt-repr-anaysis-sem/data/rte/cl_fnplus_val_lbl_file")
 sents=open("/export/ssd/apoliak/nmt-repr-anaysis-sem/data/rte/cl_fnplus_val_source_file") 
'''
