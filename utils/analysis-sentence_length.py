import itertools
import argparse
import numpy as np

def get_sent_len_stats_test():
  for dataset in ["fnplus", "sprl", "dpr"]:
    texts, hypts = [], []
    for line in open("/export/ssd/apoliak/nmt-repr-anaysis-sem/data/rte/cl_%s_val_source_file" % (dataset)):
      pair = line.split("|||")
      texts.append(len(pair[0].strip().split()))
      hypts.append(len(pair[1].strip().split()))

    diff_length = 0
    for i in range(len(texts)):
      if texts[i] != hypts[i]:
        diff_length += 1

    print "%s: diff_length: %d" % (dataset, diff_length)
    print "%s: premise:  mean sentence length: %f" % (dataset, np.mean(texts))
    print "%s: premise: number of sentences with legnth > 50: %d" % (dataset, len([i for i in texts if i > 50]))
    print "%s: premise: percent of sentences with legnth > 50: %f" % (dataset, len([i for i in texts if i > 50]) /float(len(texts)))

    print "%s: premise:  mean sentence length: %f" % (dataset, np.mean(hypts))
    print "%s: hypothesis: number of sentences with legnth > 50: %d" % (dataset, len([i for i in hypts if i > 50]))
    print "%s: hypothesis: percent of sentences with legnth > 50: %f" % (dataset, len([i for i in hypts if i > 50]) /float(len(hypts)))

def get_data(gold_f, src_f, ar_pred_f, es_pred_f, zh_pred_f, de_pred_f):

 wrong = {}
 right = {}
 for pred_f in [ar_pred_f, es_pred_f, zh_pred_f, de_pred_f]:
   gold=open(gold_f)
   sents=open(src_f)
   preds=open(pred_f)
   wrong[pred_f] = {}
   right[pred_f] = {}
   for sent, gold, pred in itertools.izip(sents, gold, pred_f): #, pred_es, pred_zh, pred_de)):
     if gold == pred:
       if gold.strip() not in right:
         right[gold.strip()] = []
       right[gold.strip()].append(sent)
     else:
       if gold.strip() not in right:
         wrong[gold.strip()] = []
       wrong[gold.strip()].append(sent)

 return right, wrong

def acc_short_sents(pred_f, gold_f, src_f, lang):
  #ar_pred = open("/export/ssd/apoliak/nmt-repr-anaysis-sem/classifiers/fnplus/UN/en-ar-4layers-brnn-enclayer4/linear_classifier/pred_file.epoch1")
  print lang
  res = {}
  for sent_lens in [0,10,20,30,40,50,60,70]: #,80]: #,90,100,110]:
    corr, tot = 0,0
    pred_es = open(pred_f)
    gold=open(gold_f)
    sents=open(src_f)
    for sent, gold, pred in itertools.izip(sents, gold, pred_es): #, pred_es, pred_zh, pred_de)):
      curr_len = len(sent.split("|||")[0].split())
      if curr_len < sent_lens:
        continue
      if curr_len >= sent_lens + 10:
        continue
      if gold.strip() == pred.strip():
        corr += 1
      tot +=1
    if tot > 0:
      res["%d-%d" % (sent_lens, sent_lens+10)] = (corr/float(tot), tot)
      print "Sentence length:%d-%d\tAccuracy: %f\t tot: %d" % (sent_lens, sent_lens+10, corr/float(tot), tot)
  return res

def get_source_sents(src_f):
  src = {}
  sents=open(src_f)
  for line in sents:
    pair = line.split("|||")
    if pair[0] not in src:
      src[pair[0]] = []
    src[pair[0]].append(pair[1])

  return src


def main(args):
  get_sent_len_stats_test()

  srcs = get_source_sents(args.src)
  '''
  right, wrong = get_data() #args.src, args.gold, args.pred)
  right_loc, wrong_loc = {}, {}

  for key in right:
    for pair in right[key]:
      t, h = pair.split("|||")[0].strip().split(), pair.split("|||")[1].strip().split()
      loc_word = 0
      if len(t) != len(h):
        continue
      for i in range(len(t)):
        if t[i] != h[i]:
          loc_word = i
      if key not in right_loc:
        right_loc[key] = []
      right_loc[key].append(loc_word/float(len(t)))

  for key in wrong:
    for pair in wrong[key]:
      t, h = pair.split("|||")[0].strip().split(), pair.split("|||")[1].strip().split()
      loc_word = 0
      if len(t) != len(h):
        continue
      for i in range(len(t)):
        if t[i] != h[i]:
          loc_word = i
      if key not in wrong_loc:
        wrong_loc[key] = []
      wrong_loc[key].append(loc_word/float(len(t)))

  print "Swapped word avg pos: %f" % np.mean(wrong_loc['entailed'] + right_loc['entailed'] + wrong_loc['not-entailed'] + right_loc['not-entailed'])

  print "Predicted entailed pos: %f" % (np.mean(wrong_loc['entailed'] + right_loc['entailed']))
  print "Predicted not-entailed pos: %f" % (np.mean(wrong_loc['not-entailed'] + right_loc['not-entailed']))

  print "Correctly predicted entailed pos: %f" % (np.mean(right_loc['entailed']))
  print "Correctly Predicted not-entailed pos: %f" % (np.mean(right_loc['not-entailed']))
  '''

  langs2stats = {}
  for pair in [(args.ar_pred, "ar"), (args.es_pred, "es"), (args.zh_pred, "zh"), (args.de_pred, "de")]: #, (args.zh_pred, "zh"), (args.es_pred, "es")]:
    tmp = acc_short_sents(pair[0], args.gold, args.src, pair[1])
    if pair[1] not in langs2stats:
      langs2stats[pair[1]] = {}
    langs2stats[pair[1]] =  tmp

  latex = ""
  keys = langs2stats[langs2stats.keys()[0]].keys()
  keys.sort()
  for key in keys:
    latex += "\small{%s} & " % (key)
    for lang in langs2stats:
      latex += " \small{%.1f} & " % (100*langs2stats[lang][key][0])
    latex += "\small{%d} \\\\ \n" % (langs2stats[lang][key][1]) 
  #latex += "\\n \\"
  print latex

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Analysis on fnplus.')
  parser.add_argument('--src', help="path to source file")
  parser.add_argument('--gold', help="path to gold labels file")
  parser.add_argument('--ar_pred', help="path to pred labels file")
  parser.add_argument('--de_pred', help="path to pred labels file")
  parser.add_argument('--zh_pred', help="path to pred labels file")
  parser.add_argument('--es_pred', help="path to pred labels file")



  args = parser.parse_args()

  main(args)

