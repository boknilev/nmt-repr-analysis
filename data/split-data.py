#_data.txt
import glob

def main():
  f_train_lbl = open("rte/cl_train_lbl_file", "wb")
  f_dev_lbl = open("rte/cl_dev_lbl_file", "wb")
  f_test_lbl = open("rte/cl_test_lbl_file", "wb")

  f_train_source = open("rte/cl_train_source_file", "wb")
  f_dev_source = open("rte/cl_dev_source_file", "wb")
  f_test_source = open("rte/cl_test_source_file", "wb")

  out_files = {"train": [f_train_lbl, f_train_source], "dev": [f_dev_lbl, f_dev_source], "test": [f_test_lbl, f_test_source]}

  input_files = glob.glob("./rte/*_data.txt")
  for file in input_files:
    orig_sent, hyp_sent, data_split, src, label = None, None, None, None, None
    for line in open(file):
      if line.startswith("entailed: "):
        label = 1
        if "not-entailed" in line:
          label = 0
      elif line.startswith("text: "):
        orig_sent = " ".join(line.split("text: ")[1:]).strip()
      elif line.startswith("hypothesis: "):
        hyp_sent = " ".join(line.split("hypothesis: ")[1:]).strip()
      elif line.startswith("partof: "):
        data_split = line.split("partof: ")[-1].strip()
      elif line.startswith("provenance: "):
        src = line.split("provenance: ")[-1].strip()
      elif not line.strip():
        assert orig_sent != None
        assert hyp_sent != None
        assert data_split != None
        assert src != None
        assert label != None
        #print orig_sent, hyp_sent, data_split, src, label
        out_files[data_split][0].write(str(label) +  "\n")
        out_files[data_split][1].write(orig_sent + "|||" + hyp_sent + "\n")
 
        orig_sent, hyp_sent, data_split, src, label = None, None, None, None, None

  for data_type in out_files:
    out_files[data_type][0].close()
    out_files[data_type][1].close()

if __name__ == '__main__':
  main()
