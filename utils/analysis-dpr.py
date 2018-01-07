import itertools
import pdb
import argparse

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


def main(args):
  data = get_data(args) #args.src, args.gold, args.pred)

  she_sents = get_sents_by_word(data, "she")
  print [data[i][1] for i in she_sents]
  pdb.set_trace()

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Analysis on dpr.')
  parser.add_argument('--src', help="path to source file")
  parser.add_argument('--gold', help="path to gold labels file")
  parser.add_argument('--pred', help="path to pred labels file")

  args = parser.parse_args()

  main(args)

#python analysis-dpr.py --src /export/ssd/apoliak/nmt-repr-anaysis-sem/data/rte/cl_dpr_val_source_file --gold /export/ssd/apoliak/nmt-repr-anaysis-sem/data/rte/cl_dpr_val_lbl_file --pred /export/ssd/apoliak/nmt-repr-anaysis-sem/output/dpr-dpr/linear_classifier/en-de/pred_file.epoch2 
