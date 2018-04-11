import argparse
import csv
import pdb
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

    print context_pos[diff_loc][-1][0],  hyp_pos[diff_loc][-1][0]
    if context_pos[diff_loc][-1][0] ==  hyp_pos[diff_loc][-1][0]:
      example_loc_to_grammar[key] = 1 #if context_pos[diff_loc][-1] ==  hyp_pos[diff_loc][-1] else 0
    else:
      example_loc_to_grammar[key] = 0

  nlp.close()

  print "Same sents count: %d" % (same_sents_count)
  pdb.set_trace()
  return example_loc_to_grammar

def main(args):
  instances = get_source(args.src)
  print "got instances"
  example_loc_to_grammar = split_data_by_grammar(instances, args.corenlp_path)

  pdb.set_trace()  


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Analysis based on POS tags in FN+.')
  parser.add_argument('--src', help="path to source file")
  parser.add_argument('--gold', help="path to gold labels file")
  parser.add_argument('--ar_pred', help="path to pred labels file")
  parser.add_argument('--de_pred', help="path to pred labels file")
  parser.add_argument('--zh_pred', help="path to pred labels file")
  parser.add_argument('--es_pred', help="path to pred labels file")
  parser.add_argument('--corenlp_path', help="path to CoreNLP directory")

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
