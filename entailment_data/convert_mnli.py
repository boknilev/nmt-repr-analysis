import sys
import csv 

#https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
csv.field_size_limit(sys.maxsize)

total_skipped = 0
label_set = set(["entailment","neutral","contradiction", "hidden"])
for f in ["train", "dev_matched", "dev_mismatched", "test_matched_unlabeled", "test_mismatched_unlabeled"]:
  version = "1.0"
  if "test" in f:
    version = "0.9"
  with open("multinli_1.0/multinli_%s_%s.txt" % (version, f), "rb") as tsvin:
    tsvin = csv.reader(tsvin, delimiter='\t', quoting=csv.QUOTE_NONE)
  
    lbl_out = open("multinli_1.0/cl_multinli_%s_lbl_file" % (f), "wb")
    source_out = open("multinli_1.0/cl_multinli_%s_source_file" % (f), "wb")

    line_num = -1
    header_size = 0
    for row in tsvin:
      line_num += 1
      if line_num == 0:
       header_size = len(row)
       continue

      if len(row) != header_size:
        print "Skipping line number %d in %s" % (line_num,f)
        total_skipped += 1
        continue
      
      lbl, sent1, sent2 = row[0].strip(), row[5].strip(), row[6].strip()
      if lbl not in label_set:
        print "Skipping line number %d in %s because of label: %s" % (line_num,f, lbl)
        total_skipped += 1
        continue

      lbl_out.write(lbl.strip() + "\n")
      source_out.write(sent1.strip() + "|||" + sent2.strip() + "\n")
  lbl_out.close()
  source_out.close()

print "Skipped a total of %d sentences: " % (total_skipped)
