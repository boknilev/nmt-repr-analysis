# Analyzing Neural MT Representations



## Instructions

Run `th src/train.lua` with the following mandatory arguments:

* `model`: NMT model trained with [seq2seq-attn](https://github.com/harvardnlp/seq2seq-attn)
* `src_dict`: source dictionary used for training the NMT model
* `targ_dict`: target dictionary used for training the NMT model
* `cl_train_lbl_file`: train file with gold labels
* `cl_val_lbl_file`: validation file with gold labels
* `cl_test_lbl_file`: test file with gold labels
* `cl_train_source_file`: train file with sentences in the source language
* `cl_val_source_file`: validation file with sentences in the source language
* `cl_test_source_file`: test file with sentences in the source language
* `cl_save`: Path to folder where experiment will be saved
* `cl_pred_file`: Prefix to save prediction files (should be base name, not full path)

Files with labels and sentences (`cl_train/val/test_lbl/source_file`) should have one sentence per line.

For more options, see `src/s2sa/beam.lua`

### Evaluating saved models
Once you've run `train.lua` and have some models saved, you can run `th src/eval.lua` with the following arguments to evaluate a model:
* `model`: NMT model trained with [seq2seq-attn](https://github.com/harvardnlp/seq2seq-attn) to extract embeddings
* `src_dict`: source dictionary used for training the NMT model
* `targ_dict`: target dictionary used for training the NMT model
* `cl_clf_model`: pretrained classifier model to load (trained with `train.lua`)
* `cl_train_lbl_file`: train file with gold labels (Used to extract labels and create label-index mapping)
* `cl_test_lbl_file`: test file with gold labels
* `cl_test_source_file`: test file with sentences in the source language
* `cl_save`: Path where the test logs will be saved
* `cl_pred_file`: Prefix to save prediction files (should be base name, not full path)

### Acknowledgements 
This project uses code from [seq2seq-attn](https://github.com/harvardnlp/seq2seq-attn).
