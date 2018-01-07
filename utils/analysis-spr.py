import itertools
import pdb
import argparse

PROTO_ROLES = set(["sentient", "aware of being involved", "existed after", "existed before", "existed during", "chose to be involved", "changed possession", "describes the location", " stationary during", "made physical contact with someone or something", "used in", "caused a change", "changes location during", "altered or somehow changed during", "existed as a physical object", "caused the", "used in carrying out"])

def get_data(args):
 lbls_file = open(args.gold)
 src_file = open(args.src)
 pred_file = open(args.pred)

 idx = {}
 for i, trip in enumerate(itertools.izip(lbls_file, pred_file, src_file)):
   idx[i] = trip[0], trip[1], trip[2]
 
 return idx

def get_idx_by_role(data):
  role2idx = {}
  for role in PROTO_ROLES:
    role2idx[role] = []
  for idx in data:
    found_role = False
    hyp = data[idx][2].split("|||")[-1]
    for role in PROTO_ROLES:
      if role in hyp:
        found_role = True
        role2idx[role].append(idx)
    if not found_role:
      print hyp    
  return role2idx

def main(args):
  data = get_data(args) #args.src, args.gold, args.pred)

  role2idx = get_idx_by_role(data)

  for role, locs in role2idx.items():
    role_tot, role_corr = 0.0,0
    for loc in locs:
      if data[loc][0] == data[loc][1]:
        role_corr += 1
      role_tot += 1.0
    print "%s\t%f\t%d" % (role, role_corr/role_tot, role_tot)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Analysis on dpr.')
  parser.add_argument('--src', help="path to source file")
  parser.add_argument('--gold', help="path to gold labels file")
  parser.add_argument('--pred', help="path to pred labels file")

  args = parser.parse_args()

  main(args)

#python analysis-dpr.py --src /export/ssd/apoliak/nmt-repr-anaysis-sem/data/rte/cl_dpr_val_source_file --gold /export/ssd/apoliak/nmt-repr-anaysis-sem/data/rte/cl_dpr_val_lbl_file --pred /export/ssd/apoliak/nmt-repr-anaysis-sem/output/dpr-dpr/linear_classifier/en-de/pred_file.epoch2 
