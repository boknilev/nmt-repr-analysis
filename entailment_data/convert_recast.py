#_data.txt
import glob

def main():
  train_count = 0
  val_count = 0
  test_count = 0
  input_files = glob.glob("./rte/*_data.txt")

  f_train_orig_data = open("rte/cl_train_orig_dataset_file", "wb")
  f_val_orig_data = open("rte/cl_val_orig_dataset_file", "wb")
  f_test_orig_data = open("rte/cl_test_orig_dataset_file", "wb")
  for file in input_files:

    f_type = file.split("/")[-1].split("_")[0] 
    f_train_lbl = open("rte/cl_"+f_type+"_train_lbl_file", "wb")
    f_dev_lbl = open("rte/cl_"+f_type+"_val_lbl_file", "wb")
    f_test_lbl = open("rte/cl_"+f_type+"_test_lbl_file", "wb")

    f_train_source = open("rte/cl_"+f_type+"_train_source_file", "wb")
    f_dev_source = open("rte/cl_"+f_type+"_val_source_file", "wb")
    f_test_source = open("rte/cl_"+f_type+"_test_source_file", "wb")


    out_files = {"train": [f_train_lbl, f_train_source, f_train_orig_data], \
                "dev": [f_dev_lbl, f_dev_source, f_val_orig_data], \
               "test": [f_test_lbl, f_test_source, f_test_orig_data]}


    orig_sent, hyp_sent, data_split, src, label = None, None, None, None, None
    for line in open(file):
      if line.startswith("entailed: "):
        label = "entailed"
        if "not-entailed" in line:
          label = "not-entailed"
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
        '''if data_split == 'train':  and train_count > 1000:
          continue
        elif data_split == 'train':
          train_count += 1
        elif data_split == 'dev' and val_count > 1000:
          continue
        elif data_split == 'dev':
          val_count += 1 
        elif data_split == 'test' and test_count > 100:
          continue
        elif data_split == 'test':
          test_count += 1
        ''' 
        #print orig_sent, hyp_sent, data_split, src, label
        out_files[data_split][0].write(str(label) +  "\n")
        out_files[data_split][1].write(orig_sent + "|||" + hyp_sent + "\n")
        out_files[data_split][2].write(file.split("/")[-1].split("_")[0] + "\n")

        orig_sent, hyp_sent, data_split, src, label = None, None, None, None, None

  for data_type in out_files:
    out_files[data_type][0].close()
    out_files[data_type][1].close()

if __name__ == '__main__':
  main()
