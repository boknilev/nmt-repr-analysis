import pandas as pd

for f in ["train", "dev", "test"]:
  df = pd.read_table("snli_1.0/snli_1.0_%s.txt" % (f))
  if f == "dev":
    f = "val"

  sentence_ones = df['sentence1']
  sentence_twos = df['sentence2']
  labels = df['gold_label']
  assert(len(labels) == len(sentence_ones) == len(sentence_twos))
  lbl_out = open("snli_1.0/cl_snli_%s_lbl_file" % (f), "wb")
  source_out = open("snli_1.0/cl_snli_%s_source_file" % (f), "wb")
  label_set = set(["entailment","neutral","contradiction"])
  for i in range(len(labels)):
    if labels[i] not in label_set:
      continue
    try:
      if sentence_twos[i].isdigit() or sentence_ones[i].isdigit():
        continue
      lbl_out.write(labels[i].strip() + "\n")
      source_out.write(sentence_ones[i].strip() + "|||" + sentence_twos[i].strip() + "\n")
    except:
      continue
      # There are a lot of examples where only the premise sentence was given
      # The sentence is often something like: cannot see the picture

  lbl_out.close()
  source_out.close()
