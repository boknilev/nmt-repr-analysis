import itertools
import pdb
import argparse

PRONOUNS = set(("he", "they", "she", "it", "him", "her"))

def get_data(args):
 lbls_file = open(args.gold)
 src_file = open(args.src)
 pred_file = open(args.pred)

 idx = {}
 for i, trip in enumerate(itertools.izip(lbls_file, pred_file, src_file)):
   idx[i] = trip[0], trip[1], trip[2]
 
 return idx

def get_sents_by_word(data, word):
  ids = []
  for key,val in data.items():
    if word in set(val[2].split()):
      ids.append(key)

  return ids

def get_pair_by_pronoun(data):
  pronoun_to_lbl = {}
  total_pairs = 0
  missed_pronouns = set()
  for pronoun in PRONOUNS:
    pronoun_to_lbl[pronoun] = {'correct': {'entailed': [], 'not-entailed': []}, 'incorrect': {'entailed': [], 'not-entailed': []} }
    for idx, trip in data.items():
      src = trip[2].split("|||")
      context = src[0]
      hyp = src[1]
      if pronoun in context.lower() and pronoun not in hyp.lower():
        total_pairs += 1
        gold = trip[0].strip()
        pred = trip[1].strip()
        if gold == pred:
          pronoun_to_lbl[pronoun]['correct'][gold].append(trip[2])
        else:
          #incorrect is based off of the gold label, not pred label
          pronoun_to_lbl[pronoun]['incorrect'][gold].append(trip[2])
      else:
        set_diff = set(context.split()).difference(set(hyp.split())) 
        assert (len(set_diff) == 1, "more than one word difference")
        word = list(set_diff)[0]
        if word not in PRONOUNS:
          missed_pronouns.add(word)

  return pronoun_to_lbl

def main(args):
  data = get_data(args) #args.src, args.gold, args.pred)

  pronoun_to_lbl = get_pair_by_pronoun(data)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Analysis on dpr.')
  parser.add_argument('--src', help="path to source file")
  parser.add_argument('--gold', help="path to gold labels file")
  parser.add_argument('--pred', help="path to pred labels file")

  args = parser.parse_args()

  main(args)

#python analysis-dpr.py --src /export/ssd/apoliak/nmt-repr-anaysis-sem/data/rte/cl_dpr_val_source_file --gold /export/ssd/apoliak/nmt-repr-anaysis-sem/data/rte/cl_dpr_val_lbl_file --pred /export/ssd/apoliak/nmt-repr-anaysis-sem/output/dpr-dpr/linear_classifier/en-de/pred_file.epoch2 
