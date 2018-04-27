# Analyzing Neural MT Representations

This repository contains the code for our paper on analyzing morphology in neural machine translation models:

"What do Neural Machine Translation Models Learn about Morphology?", Yonatan Belinkov, Nadir Durrani, Fahim Dalvi, Hassan Sajjad, and James Glass, ACL 2017. 

## Requirements

Torch - [http://torch.ch/docs/getting-started.html](http://torch.ch/docs/getting-started.html)

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
* `cl_enc_layer`: Which layer in the encoder/decoder to use for word representation (0 for word embeddings)

### Optional command line arguments
Here are some optional command line arguments that are useful:

* `cl_avg_reps`: Whether to average the hidden representations states for the sentences. Otherwise we use the right-most hidden state from the forward LSTM encoder and the left most from the backwards LSTM encoder.
* `cl_entailment`: Whether to train the classifier to classifier sentence pairs for textual entailment
* `cl_enc_layer`: Which hidden layer to extract the hidden states from. This command line argument needs to be followed by an integer. By defualt, the argument is set to 2. Warning: the integer must be less than or equal to the number of hidden states in the NMT model, otherwise you will run into a error related to indexing out of bounds.
* `cl_inferSent_reps`: Whether to use the sentence representations as described in Conneau (EMNLP '17)'s [InferSet](https://arxiv.org/pdf/1705.02364.pdf), otherwise the first and last hidden representations will be used and then the sentence reps. will be concatenated.

## Citing
If you use this code in your work, please consider citing our paper:


```
@InProceedings{belinkov:2017:acl,
  author    = {Belinkov, Yonatan  and  Durrani, Nadir and Dalvi, Fahim and Sajjad, Hassan and Glass, James},
  title     = {What do Neural Machine Translation Models Learn about Morphology?},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL)},
  month     = {July},
  year      = {2017},
  address   = {Vancouver},
  publisher = {Association for Computational Linguistics},
}
```

If you use this code for NLI please cite the following paper as well:

```
@inproceedings{evaluating-fine-grained-semantic-phenomena-in-neural-machine-translation-encoders-using-entailment,
  author = {Poliak, Adam and Belinkov, Yonatan and Glass, James and {Van Durme}, Benjamin},
  title = {On the Evaluation of Semantic Phenomena in Neural Machine Translation Using Natural Language Inference},
  year = {2018},
  booktitle = {Proceedings of the Annual Meeting of the North American Association of Computational Linguistics (NAACL)}
}
```

### Acknowledgements 
This project uses code from [seq2seq-attn](https://github.com/harvardnlp/seq2seq-attn).
