import itertools
import pdb
import argparse

PROTO_ROLES = set(["sentient", "aware of being involved", "existed after", "existed before", "existed during", "chose to be involved", "changed possession", "describes the location", "stationary during", "made physical contact with someone or something", "caused a change", "changes location during", "altered or somehow changed during", "existed as a physical object", "caused the", "used in carrying out"])

role2str = {"sentient" : "sentient", "aware of being involved" : "aware", "existed after" : "existed after", \
            "existed before" : "existed before", "existed during" : "existed during", "chose to be involved" : "volitional", \
            "changed possession" : "chang. possession", "describes the location": "location", "stationary during" : "stationary during" , \
            "made physical contact with someone or something" : "physical contact", "caused a change" : "changed", \
            "changes location during" : "moved", "altered or somehow changed during" : "changed", \
            "existed as a physical object" : "physically existed", "caused the" : "caused", "used in carrying out" : "used in"}


AGENT_ROLES = ["existed as a physical object", "sentient", "aware of being involved", "chose to be involved", \
                "existed before", "caused the"]

PATIENT_ROLES = [key for key in role2str if key not in AGENT_ROLES]

def get_data(args):
 lbls_file = open(args.gold)
 src_file = open(args.src)
 ar_pred_file = open(args.ar_pred)
 es_pred_file = open(args.es_pred)
 zh_pred_file = open(args.zh_pred)
 de_pred_file = open(args.de_pred)

 idx = {}
 for i, trip in enumerate(itertools.izip(lbls_file, src_file, ar_pred_file, es_pred_file, zh_pred_file, de_pred_file)):
   idx[i] = trip #trip[0], trip[1], trip[2], trip[3], trip[4]
 
 return idx

def get_idx_by_role(data):
  role2idx = {}
  for role in PROTO_ROLES:
    role2idx[role] = []
  for idx in data:
    found_role = False
    hyp = data[idx][1].split("|||")[-1]
    for role in PROTO_ROLES:
      if role in hyp:
        found_role = True
        role2idx[role].append(idx)
    if not found_role:
      print hyp    
  return role2idx

def main(args):
  data = get_data(args) #args.src, args.gold, args.pred)

  maj_entailed = 0.0

  role2idx = get_idx_by_role(data)

  role2pos_count, role2not_count = {}, {}
  role2lang2stats = {} # stat is [entailed_correct, not_correct]
  for role in role2idx:
    role2lang2stats[role] = {'ar': [0.0, 0.0], 'es': [0.0, 0.0], 'zh': [0.0, 0.0], 'de': [0.0, 0.0]}
  print "\small{%s}\t& %s\t& %s\t& %s \t& %s\t& %s & & & %s\\\\ \\hline" % ("Proto-Role", "ar", "es", "zh", "de", "avg", "MAJ")

  for role in AGENT_ROLES + PATIENT_ROLES:
    locs = role2idx[role]
    if role not in role2pos_count:
      role2pos_count[role] = 0.0
    if role not in role2not_count:
      role2not_count[role] = 0.0
    role_tot, ar_corr, es_corr, zh_corr, de_corr = 0.0, 0, 0, 0, 0
    for loc in locs:
      if "not" not in data[loc][0]:
        role2pos_count[role] += 1
      else:
          role2not_count[role] += 1
      if data[loc][0] == data[loc][2]:
        ar_corr += 1
        if "not" not in data[loc][0]:
          role2lang2stats[role]['ar'][0] += 1
        else:
          role2lang2stats[role]['ar'][1] += 1
      if data[loc][0] == data[loc][3]:
        es_corr += 1
        if "not" not in data[loc][0]:
          role2lang2stats[role]['es'][0] += 1
        else:
          role2lang2stats[role]['es'][1] += 1
      if data[loc][0] == data[loc][4]:
        zh_corr += 1
        if "not" not in data[loc][0]:
          role2lang2stats[role]['zh'][0] += 1
        else:
          role2lang2stats[role]['zh'][1] += 1
      if data[loc][0] == data[loc][5]:
        de_corr += 1 
        if "not" not in data[loc][0]:
          role2lang2stats[role]['de'][0] += 1
        else:
          role2lang2stats[role]['de'][1] += 1
      role_tot += 1.0
    print "\small{%s}\t& %.1f\t& %.1f\t& %.1f \t& %.1f\t& %.1f\t& & & %.1f \\\\" % (role2str[role], 100*ar_corr/role_tot, 100*es_corr/role_tot, 100*zh_corr/role_tot, 100*de_corr/role_tot, \
                                     100*(ar_corr + es_corr + zh_corr + de_corr)/ (4 * role_tot), 100*(max(1 - (role2pos_count[role]/role_tot), (role2pos_count[role]/role_tot))))
    if 1 - (role2pos_count[role]/role_tot) <  (role2pos_count[role]/role_tot):
      maj_entailed += 1
    if role == "caused the":
      print "\hline"

  print "For %.2f percent of the roles, the majority label was entailed." % (100 * maj_entailed / len(role2idx))
  print

  print "\small{%s}\t& %s\t& %s\t& %s\t& %s\t& %s\t& %s\t& %s \t& %s\t& %s\t& %s\\\\ \\hline" % ("Proto-Role", "ar ent", "ar not", "es ent", "es not", "zh ent", "zh not", "de ent", "de not", "avg ent", "avg not")

  for role in AGENT_ROLES + PATIENT_ROLES:
    print "\small{%s}\t& %.1f\t& %.1f\t& %.1f\t& %.1f\t& %.1f\t& %.1f\t& %.1f\t& %.1f\t& %.1f\t& %.1f \\\\" % \
            (role2str[role], 100*role2lang2stats[role]['ar'][0]/role2pos_count[role], 100*role2lang2stats[role]['ar'][1]/role2not_count[role], \
            100*role2lang2stats[role]['es'][0]/role2pos_count[role], 100*role2lang2stats[role]['es'][1]/role2not_count[role], \
            100*role2lang2stats[role]['zh'][0]/role2pos_count[role], 100*role2lang2stats[role]['zh'][1]/role2not_count[role], \
            100*role2lang2stats[role]['de'][0]/role2pos_count[role], 100*role2lang2stats[role]['de'][1]/role2not_count[role], \
            100*(role2lang2stats[role]['ar'][0]+role2lang2stats[role]['es'][0]+role2lang2stats[role]['zh'][0]+role2lang2stats[role]['de'][0])/(4 * role2pos_count[role]), \
            100*(role2lang2stats[role]['ar'][1]+role2lang2stats[role]['es'][1]+role2lang2stats[role]['zh'][1]+role2lang2stats[role]['de'][1])/(4 * role2not_count[role]))

    if role == "caused the":
      print "\hline"

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Analysis on spr.')
  parser.add_argument('--src', help="path to source file")
  parser.add_argument('--gold', help="path to gold labels file")
  parser.add_argument('--ar_pred', help="path to en-ar pred labels file")
  parser.add_argument('--es_pred', help="path to en-es pred labels file")
  parser.add_argument('--zh_pred', help="path to en-zh pred labels file")
  parser.add_argument('--de_pred', help="path to en-de pred labels file")

  args = parser.parse_args()

  main(args)

#python analysis-spr.py --src /export/ssd/apoliak/nmt-repr-anaysis-sem/data/rte/cl_sprl_val_source_file --gold /export/ssd/apoliak/nmt-repr-anaysis-sem/data/rte/cl_sprl_val_lbl_file --de_pred /export/ssd/apoliak/nmt-repr-anaysis-sem/output/sprl-sprl/linear_classifier/en-de/test_pred_file.epoch2 --ar_pred /export/ssd/apoliak/nmt-repr-anaysis-sem/output/sprl-sprl/linear_classifier/en-de/test_pred_file.epoch2 --es_pred /export/ssd/apoliak/nmt-repr-anaysis-sem/output/sprl-sprl/linear_classifier/en-de/test_pred_file.epoch1 --zh_pred /export/ssd/apoliak/nmt-repr-anaysis-sem/output/sprl-sprl/linear_classifier/en-de/test_pred_file.epoch1
