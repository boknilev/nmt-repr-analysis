-- uses code from: https://github.com/harvardnlp/seq2seq-attn

local beam = require 's2sa.beam'
require 'nn'
require 'xlua'
require 'optim'
seq = require 'pl.seq'
stringx = require 'pl.stringx'

function main()
  print(arg)
  beam.init(arg)
  opt = beam.getOptions()
  skip_start_end = model_opt.start_symbol == 1
  --print('start_symbol: ' .. model_opt.start_symbol)
  --print('skip_start_end:')
  --print(skip_start_end)
  
  classifier_opt = get_classifier_options(opt)
  classifier_opt.pred_file = paths.concat(classifier_opt.save, classifier_opt.pred_file)
  if classifier_opt.classifier_size == 0 then classifier_opt.classifier_size = model_opt.rnn_size end  
  
  assert(path.exists(classifier_opt.train_lbl_file), 'train_lbl_file does not exist')
  assert(path.exists(classifier_opt.test_lbl_file), 'test_lbl_file does not exist')
  assert(path.exists(classifier_opt.test_source_file), 'test_source_file does not exist')
  if classifier_opt.enc_or_dec == 'dec' then
    assert(path.exists(classifier_opt.train_target_file), 'train_target_file does not exist')
    assert(path.exists(classifier_opt.test_target_file), 'test_target_file does not exist')    
  end
  assert(path.exists(classifier_opt.save), 'save dir does not exist')
  assert(path.exists(classifier_opt.clf_model), 'classifier model does not exist')
  
  -- number of module for word representation
  module_num = 2*classifier_opt.enc_layer - classifier_opt.use_cell
 
  -- first pass: get labels
  print('==> first pass: getting labels')
  label2idx, idx2label = get_labels(classifier_opt.train_lbl_file)
  local classes = {}
  for idx, _ in ipairs(idx2label) do
    table.insert(classes, idx)
  end
  classifier_opt.num_classes = #idx2label
  print('label2idx:')
  print(label2idx)
  print('idx2label:')
  print(idx2label)
  print('classes:')
  print(classes)
 
  -- second pass: prepare data as vectors
  print('==> second pass: loading data')
  test_data = load_data(classifier_opt, label2idx)
  print('model_opt.brnn: ' .. model_opt.brnn) 
  -- use trained encoder/decoder from MT model
  encoder, decoder = model[1], model[2]
  if model_opt.brnn == 1 then
    encoder_brnn = model[4]
  end
  
  -- define classifier
  classifier = torch.load(classifier_opt.clf_model)  
  print('==> loaded classification model:')
  print(classifier)

  -- define classification criterion
  criterion = nn.CrossEntropyCriterion()

  -- move to cuda
  if opt.gpuid >= 0 then     
    classifier = classifier:cuda()
    criterion = criterion:cuda()
  end

  confusion = optim.ConfusionMatrix(classes)

  -- Log results to files
  test_logger = optim.Logger(paths.concat(classifier_opt.save, 'test.log'), classifier_opt.pred_file)  
  
  collectgarbage(); collectgarbage();
  
  -- do epochs
  epoch = 1
  local test_loss
  if classifier_opt.entailment then
    test_loss = eval_entailment(test_data, epoch, test_logger, 'test', classifier_opt.pred_file)
  else
    test_loss = eval(test_data, epoch, test_logger, 'test', classifier_opt.pred_file)
  end
  collectgarbage(); collectgarbage();
end

function eval(data, epoch, logger, test_or_val, pred_filename)
  test_or_val = test_or_val or 'test'
  local pred_file
  if pred_filename then
    pred_file = torch.DiskFile(pred_filename .. '.epoch' .. epoch, 'w')
  end
  
  local time = sys.clock()
  classifier:evaluate()
  encoder:evaluate(); decoder();
  if model_opt.brnn == 1 then encoder_brnn:evaluate() end
  
  print('\n==> evaluating on ' .. test_or_val .. ' data')
  print('==> epoch: ' .. epoch)
  local loss, num_words = 0, 0
  for i=1,#data do 
    xlua.progress(i, #data)
    local source, target, labels = data[i][1]
    if opt.gpuid >= 0 then source = source:cuda() end
    if classifier_opt.enc_or_dec == 'enc' then
      labels = data[i][2]      
    else
      target = data[i][2]
      if opt.gpuid >= 0 then target = target:cuda() end
      labels = data[i][3]
    end
    
    local source_l = math.min(source:size(1), opt.max_sent_l)
    local source_input
    if model_opt.use_chars_enc == 1 then
      source_input = source:view(source_l, 1, source:size(2)):contiguous()
    else
      source_input = source:view(source_l, 1)
    end

    local rnn_state_enc = {}
    for i = 1, #init_fwd_enc do
      table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
    end
    local context = context_proto[{{}, {1,source_l}}]:clone() -- 1 x source_l x rnn_size
    -- special case when using word vectors
    if classifier_opt.enc_layer == 0 then
      if model_opt.use_chars_enc == 0 then
        context = context_proto_word_vecs[{ {}, {1,source_l}, {} }]:clone()
      else
        context = context_proto_char_cnn[{ {}, {1,source_l}, {} }]:clone()
      end
    end
    
    -- forward encoder
    for t = 1, source_l do
      -- run through encoder if using representations above word vectors or if need it for decoder
      local enc_out
      if classifier_opt.enc_layer > 0 or classifier_opt.enc_or_dec == 'dec' then
        local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
        enc_out = encoder:forward(encoder_input)
        rnn_state_enc = enc_out
      end
      if classifier_opt.enc_layer > 0 then
        context[{{},t}]:copy(enc_out[module_num])
      else
        -- TODO make sure size of context is same dimension as size of word vectors (by default they are both 500)
        local word_vec_out
        if model_opt.use_chars_enc == 0 then
          if classifier_opt.verbose then print('forwarding word_vecs_enc at t: ' .. t) end
          word_vec_out = word_vecs_enc:forward(source_input[t])
        else
          -- TODO make sure size of context is same dimension as size of charCNN output (by default the context is 500 but the charcNN is 1000)
          if classifier_opt.verbose then
            print('forwarding char_cnn_enc at t: ' .. t)
            print('source_input[t]:'); print(source_input[t]);
            print('word_vec_enc:forward(source_input[t]):'); print(word_vecs_enc:forward(source_input[t]));
            print('char_cnn_enc:forward(word_vec_enc:forward(source_input[t])):'); print(char_cnn_enc:forward(word_vecs_enc:forward(source_input[t])))
            print('mlp_enc:forward(char_cnn_enc:forward(word_vec_enc:forward(source_input[t]))):'); print(mlp_enc:forward(char_cnn_enc:forward(word_vecs_enc:forward(source_input[t]))))
          end
          word_vec_out = word_vecs_enc:forward(source_input[t])
          word_vec_out = char_cnn_enc:forward(word_vec_out)
          if model_opt.num_highway_layers > 0 then
            word_vec_out = mlp_enc:forward(word_vec_out)
          end
        end
        context[{{},t}]:copy(word_vec_out)
      end
    end
    
    local rnn_state_dec = {}
    for i = 1, #init_fwd_dec do
      table.insert(rnn_state_dec, init_fwd_dec[i]:zero())
    end
    
    if model_opt.init_dec == 1 then
      for L = 1, model_opt.num_layers do
        rnn_state_dec[L*2-1+model_opt.input_feed]:copy(rnn_state_enc[L*2-1])
        rnn_state_dec[L*2+model_opt.input_feed]:copy(rnn_state_enc[L*2])
      end
    end                    
    
    if model_opt.brnn == 1 then
      for i = 1, #rnn_state_enc do
        rnn_state_enc[i]:zero()
      end
      -- forward backward encoder
      for t = source_l, 1, -1 do
        local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
        local enc_out = encoder_brnn:forward(encoder_input)
        rnn_state_enc = enc_out
        context[{{},t}]:add(enc_out[module_num])
      end
      if model_opt.init_dec == 1 then
        for L = 1, model_opt.num_layers do
          rnn_state_dec[L*2-1+model_opt.input_feed]:add(rnn_state_enc[L*2-1])
          rnn_state_dec[L*2+model_opt.input_feed]:add(rnn_state_enc[L*2])
        end
      end                
    end
    
    local dec_all_out, target_l
    if classifier_opt.enc_or_dec == 'dec' then
      target_l = math.min(target:size(1), opt.max_sent_l)
      dec_all_out = context_proto[{{}, {1,target_l}}]:clone() 
      -- forward decoder
      for t = 2, target_l do 
        local decoder_input1
        if model_opt.use_chars_dec == 1 then
          decoder_input1 = word2charidx_targ:index(1, target[{{t-1}}]:long())
        else
          decoder_input1 = target[{{t-1}}]
        end
        local decoder_input
        if model_opt.attn == 1 then
          decoder_input = {decoder_input1, context[{{1}}], table.unpack(rnn_state_dec)}
        else
          decoder_input = {decoder_input1, context[{{1}, source_l}], table.unpack(rnn_state_dec)}
        end
        local out_decoder = decoder:forward(decoder_input)
        --local out = model[3]:forward(out_decoder[#out_decoder]) -- K x vocab_size
        rnn_state_dec = {} -- to be modified later
        if model_opt.input_feed == 1 then
          table.insert(rnn_state_dec, out_decoder[#out_decoder])
        end
        for j = 1, #out_decoder - 1 do
          table.insert(rnn_state_dec, out_decoder[j])
        end
        dec_all_out[{{},t}]:copy(out_decoder[module_num])                      
      end                            
    end  
    
    -- take encoder/decoder output as input to classifier
    local classifier_input_all
    if classifier_opt.enc_or_dec == 'dec' then
      -- always ignore start and end sybmols in dec
      local end_idx = target_l == opt.max_sent_len and target_l or target_l-1
      classifier_input_all = dec_all_out[{{}, {2,end_idx}}]
    else
      if not skip_start_end then
        classifier_input_all = context
      else
        local end_idx = source_l == opt.max_sent_len and source_l or source_l-1
        classifier_input_all = context[{{}, {2,end_idx}}]
      end
    end
    
    -- forward classifier    
    local pred_labels = {}
    for t = 1, classifier_input_all:size(2) do
      -- take word representation
      local classifier_input = classifier_input_all[{{},t}]      
      classifier_input = classifier_input:view(classifier_input:nElement())        
      local classifier_out = classifier:forward(classifier_input)
      -- get predicted labels to write to file
      if pred_file then
        local _, pred_idx =  classifier_out:max(1)
        pred_idx = pred_idx:long()[1]
        local pred_label = idx2label[pred_idx]
        table.insert(pred_labels, pred_label)
      end
      
      loss = loss + criterion:forward(classifier_out, labels[t])
      num_words = num_words + 1
      
      confusion:add(classifier_out, labels[t])
    end
    if pred_file then
      pred_file:writeString(stringx.join(' ', pred_labels) .. '\n')
    end
    
  end
  loss = loss/num_words
  
  time = (sys.clock() - time) / #data
  print('==> time to evaluate 1 sample = ' .. (time*1000) .. 'ms') 
  print('==> loss: ' .. loss)
  
  print(confusion)
  
   -- update log/plot
   logger:add{['% mean class accuracy (' .. test_or_val .. ' set)'] = confusion.totalValid * 100}
   if classifier_opt.plot then
      logger:style{['% mean class accuracy (' .. test_or_val .. ' set)'] = '-'}
      logger:plot()
   end
   
  -- next epoch
  confusion:zero()
  
  if pred_file then pred_file:close() end
  return loss
  
end

function eval_entailment(data, epoch, logger, test_or_val, pred_filename)
  test_or_val = test_or_val or 'test'
  local pred_file
  if pred_filename then
    pred_file = torch.DiskFile(pred_filename .. '.epoch' .. epoch, 'w')
  end
  local word_repr_file
  if pred_file and classifier_opt.write_test_word_repr and classifier_opt.test_word_repr_file then
    word_repr_file = torch.DiskFile(classifier_opt.test_word_repr_file, 'w')
  end

  local time = sys.clock()
  classifier:evaluate()
  encoder:evaluate(); decoder:evaluate();
  if model_opt.brnn == 1 then encoder_brnn:evaluate() end

  print('\n==> evaluating on ' .. test_or_val .. ' data')
  print('==> epoch: ' .. epoch)
  local loss, num_words, word_counter = 0, 0, 0
  for i=1,#data do
    xlua.progress(i, #data)
    local t_source = data[i][1]
    local h_source = data[i][2]
    local label = data[i][3]
    if opt.gpuid >= 0 then t_source = t_source:cuda(); h_source = h_source:cuda(); end

    local t_source_l = math.min(t_source:size(1), opt.max_sent_l)
    local h_source_l = math.min(h_source:size(1), opt.max_sent_l)
    local t_source_input, h_source_input
    if model_opt.use_chars_enc == 1 then
      t_source_input = t_source:view(t_source_l, 1, t_source:size(2)):contiguous()
      h_source_input = h_source:view(h_source_l, 1, h_source:size(2)):contiguous()
    else
      t_source_input = t_source:view(t_source_l, 1)
      h_source_input = h_source:view(h_source_l, 1)
    end

    local t_context = context_proto[{{}, {1,t_source_l}}]:clone() -- 1 x source_l x rnn_size
    local h_context = context_proto[{{}, {1,h_source_l}}]:clone() -- 1 x source_l x rnn_size

    if classifier_opt.enc_layer == 0 then
      if model_opt.use_chars_enc == 0 then
        t_context = context_proto_word_vecs[{ {}, {1,t_source_l}, {} }]:clone()
        h_context = context_proto_word_vecs[{ {}, {1,h_source_l}, {} }]:clone()
      else
        t_context = context_proto_char_cnn[{ {}, {1,t_source_l}, {} }]:clone()
        h_context = context_proto_char_cnn[{ {}, {1,h_source_l}, {} }]:clone()
      end
    end

    -- forward encoder for t sentence
    local rnn_state_enc = {}
    for i = 1, #init_fwd_enc do
      table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
    end
    local pred_labels = {}
    for t = 1, t_source_l do
      local t_enc_out
      if classifier_opt.enc_layer > 0 then
        local t_encoder_input = {t_source_input[t], table.unpack(rnn_state_enc)}
        t_enc_out = encoder:forward(t_encoder_input)
        rnn_state_enc = t_enc_out
      end
      if classifier_opt.enc_layer > 0 then
        t_context[{{},t}]:copy(t_enc_out[module_num])
      end
    end
    local t_forward
    if classifier_opt.inferSent_reps then
      t_forward = t_context:max(2)[{{}, 1}]
    elseif classifier_opt.avg_reps then
      t_forward = t_context:mean(2)[{{}, 1}]
    else
      t_forward = t_context[{{}, h_source_l}]
    end
    t_context:zero()
    if model_opt.brnn == 1 then
      for i = 1, #rnn_state_enc do
        rnn_state_enc[i]:zero()
      end
      if classifier_opt.verbose then print('forward bwd encoder') end
      for t = t_source_l, 1, -1 do
        if classifier_opt.enc_layer > 0 then
          local t_encoder_input = {t_source_input[t], table.unpack(rnn_state_enc)}
          local t_enc_out = encoder_brnn:forward(t_encoder_input)
          rnn_state_enc = t_enc_out
          t_context[{{},t}]:add(t_enc_out[module_num])
          if classifier_opt.verbose then
            print('t: ' .. t)
            print('t_encoder_input:'); print(t_encoder_input);
            print('t_enc_out:'); print(t_enc_out);
          end
        end
      end
    end

    -- forward encoder for h sentence
    local rnn_state_enc = {}
    for i = 1, #init_fwd_enc do
      table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
    end
    for t = 1, h_source_l do
      local h_enc_out
      if classifier_opt.enc_layer > 0 then
        local h_encoder_input = {h_source_input[t], table.unpack(rnn_state_enc)}
        h_enc_out = encoder:forward(h_encoder_input)
        rnn_state_enc = h_enc_out
      end
      if classifier_opt.enc_layer > 0 then
        h_context[{{},t}]:copy(h_enc_out[module_num])
      end
    end
    local h_forward
    if classifier_opt.inferSent_reps then
      h_forward = h_context:max(2)[{{}, 1}]
    elseif classifier_opt.avg_reps then
      h_forward = h_context:mean(2)[{{}, 1}]
    else
      h_forward = h_context[{{}, h_source_l}]
    end
    h_context:zero()
    if model_opt.brnn == 1 then
      for i = 1, #rnn_state_enc do
        rnn_state_enc[i]:zero()
      end
      if classifier_opt.verbose then print('forward bwd encoder') end
      for t = h_source_l, 1, -1 do
        if classifier_opt.enc_layer > 0 then
          local h_encoder_input = {h_source_input[t], table.unpack(rnn_state_enc)}
          local h_enc_out = encoder_brnn:forward(h_encoder_input)
          rnn_state_enc = h_enc_out
          h_context[{{},t}]:add(h_enc_out[module_num])
          if classifier_opt.verbose then
            print('t: ' .. t)
            print('h_encoder_input:'); print(h_encoder_input);
            print('h_enc_out:'); print(h_enc_out);
          end
        end
      end
    end

    local classifier_input
    if model_opt.brnn == 1 and classifier_opt.avg_reps then
      classifier_input = torch.cat(t_forward, t_context:mean(2)[{{}, 1}])
      classifier_input = torch.cat(classifier_input, h_forward)
      classifier_input = torch.cat(classifier_input, h_context:mean(2)[{{}, 1}])
    elseif model_opt.brnn == 1 and classifier_opt.inferSent_reps then
      local t_sent = torch.cat(t_forward, t_context:max(2)[{{}, 1}])
      local h_sent = torch.cat(h_forward, h_context:max(2)[{{}, 1}])
      classifier_input = torch.cat(t_sent, h_sent)
      classifier_input = torch.cat(classifier_input, torch.abs(t_sent - h_sent))
      classifier_input = torch.cat(classifier_input, torch.cmul(h_sent, t_sent))
    elseif model_opt.brnn == 1 then
      classifier_input = torch.cat(t_forward, t_context[{{},1}])
      classifier_input = torch.cat(classifier_input, h_forward)
      classifier_input = torch.cat(classifier_input, h_context[{{},1}])
    elseif classifier_opt.avg_reps then
      classifier_input = torch.cat(t_forward, h_forward)
    elseif classifier_opt.inferSent_reps then
      local t_sent = t_forward
      local h_sent = h_forward
      classifier_input = torch.cat(t_sent, h_sent)
      classifier_input = torch.cat(classifier_input, torch.abs(t_sent - h_sent))
      classifier_input = torch.cat(classifier_input, torch.cmul(h_sent, t_sent))
    else
      classifier_input = torch.cat(t_context[{{},t_source_l}], h_context[{{},h_source_l}])
    end
    local classifier_out = classifier:forward(classifier_input)
    if pred_file then
      local _, pred_idx =  classifier_out:transpose(1,2):max(1)
      pred_idx = pred_idx:long()[1]
      local pred_label = idx2label[pred_idx[1]]
      table.insert(pred_labels, pred_label)
    end

    loss = loss + criterion:forward(classifier_out, label)
    num_words = num_words + 1

    confusion:add(classifier_out[{1, {}}], label)

    if pred_file then
      pred_file:writeString(stringx.join(' ', pred_labels) .. '\n')
    end
  end
  loss = loss/num_words

  time = (sys.clock() - time) / #data
  print('==> time to evaluate 1 sample = ' .. (time*1000) .. 'ms')
  print('==> loss: ' .. loss)

  print(confusion)

  logger:add{['% mean class accuracy (' .. test_or_val .. ' set)'] = confusion.totalValid * 100}
  if classifier_opt.plot then
    logger:style{['% mean class accuracy (' .. test_or_val .. ' set)'] = '-'}
    logger:plot()
  end

  confusion:zero()

  if pred_file then pred_file:close() end
  if word_repr_file then word_repr_file:close() end
  return loss
end

function load_data(classifier_opt, label2idx)
  local test_data
  print(classifier_opt.entailment)
  if classifier_opt.entailment then
    unknown_labels = 0
    test_data = load_source_entailment_data(classifier_opt.test_source_file, classifier_opt.test_lbl_file, label2idx)
    print('==> words with unknown labels in train data: ' .. unknown_labels)
    assert(unknown_labels == 0, 'Test data contained an unknown label')
  elseif classifier_opt.enc_or_dec == 'enc' then
    unknown_labels = 0
    test_data = load_source_data(classifier_opt.test_source_file, classifier_opt.test_lbl_file, label2idx)   
    print('==> words with unknown labels in test data: ' .. unknown_labels)
  else
    unknown_labels = 0
    test_data = load_source_target_data(classifier_opt.test_source_file, classifier_opt.test_target_file, classifier_opt.test_lbl_file, label2idx)   
    print('==> words with unknown labels in test data: ' .. unknown_labels)    
  end
  return test_data
end

function load_source_data(file, label_file, label2idx, max_sent_len) 
  local max_sent_len = max_sent_len or math.huge
  local data = {}
  for line, labels in seq.zip(io.lines(file), io.lines(label_file)) do
    sent = beam.clean_sent(line)
    local source
    if model_opt.use_chars_enc == 0 then
      source, _ = beam.sent2wordidx(line, word2idx_src, model_opt.start_symbol)
    else
      source, _ = beam.sent2charidx(line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
    end    
    local label_idx = {}
    for label in labels:gmatch'([^%s]+)' do
      if label2idx[label] then
        idx = label2idx[label]
      else
        print('Warning: unknown label ' .. label .. ' in line: ' .. line .. ' with labels ' .. labels)
        print('Warning: using idx 0 for unknown')
        idx = 0
        unknown_labels = unknown_labels + 1
      end
      table.insert(label_idx, idx)
    end
    if #label_idx <= max_sent_len then
      table.insert(data, {source, label_idx})
    end
  end
  return data
end

function load_source_entailment_data(file, label_file, label2idx, max_sent_len)
  local max_sent_len = max_sent_len or math.huge
  data = {}
  for sents, label in seq.zip2(io.lines(file), io.lines(label_file)) do
    sent_list = beam.clean_sents(sents)
    t_sent = sent_list[1]
    h_sent = sent_list[2]
    local t_source, h_source
    if model_opt.use_chars_enc == 0 then
      t_source, _ = beam.sent2wordidx(t_sent, word2idx_src, model_opt.start_symbol)
      h_source, _ = beam.sent2wordidx(h_sent, word2idx_src, model_opt.start_symbol)
    else
      t_source, _ = beam.sent2charidx(t_sent, char2idx, model_opt.max_word_l, model_opt.start_symbol)
      h_source, _ = beam.sent2charidx(h_sent, char2idx, model_opt.max_word_l, model_opt.start_symbol)
    end
    if t_source:dim() == 0 then
      print('Warning: empty source vector in test sentence ' .. t_sent)
    end
    if h_source:dim() == 0 then
      print('Warning: empty source vector in hypothesis sentence ' .. h_sent)
    end
    --h_l, t_l = #stringx.split(h_sent, " "), #stringx.split(t_sent, " ")
    --if h_l <= max_sent_len and h_l >= 1 and t_l <= max_sent_len and t_l >= 1 then
    table.insert(data, {t_source, h_source, label2idx[label]})
    --end
  end
  return data
end

function load_source_target_data(source_file, target_file, target_label_file, label2idx, max_sent_len)
  local max_sent_len = max_sent_len or math.huge
  local data = {}
  for source_line, target_line, labels in seq.zip3(io.lines(source_file), io.lines(target_file), io.lines(target_label_file)) do
    source_sent = beam.clean_sent(source_line)
    local source
    if model_opt.use_chars_enc == 0 then
      source, _ = beam.sent2wordidx(source_line, word2idx_src, model_opt.start_symbol)
    else
      source, _ = beam.sent2charidx(source_line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
    end
    
    target_sent = beam.clean_sent(target_line)
    local target
    -- TODO make sure it's correct to always use start_symbol here (last argument) -> check the effect during training, maybe need to ignore this symbol
    if model_opt.use_chars_dec == 0 then
      target, _ = beam.sent2wordidx(target_line, word2idx_targ, 1)
    else
      --target, _ = beam.sent2charidx(target_line, char2idx, model_opt.max_word_l, 1)
      target, _ = beam.sent2wordidx(target_line, word2idx_targ, 1)
    end
    
    local label_idx = {}
    for label in labels:gmatch'([^%s]+)' do
      if label2idx[label] then
        idx = label2idx[label]
      else
        print('Warning: unknown label ' .. label .. ' in target line: ' .. target_line .. ' with labels ' .. labels)
        print('Warning: using idx 0 for unknown')
        idx = 0
        unknown_labels = unknown_labels + 1
      end
      table.insert(label_idx, idx)
    end
    if #label_idx <= max_sent_len then
      table.insert(data, {source, target, label_idx})
    end
  end
  return data

end


function get_labels(label_file)
  local label2idx, idx2label = {}, {}
  for line in io.lines(label_file) do
    for label in line:gmatch'([^%s]+)' do
      if not label2idx[label] then
        idx2label[#idx2label+1] = label
        label2idx[label] = #idx2label
      end
    end
  end
  return label2idx, idx2label
end

function seq.zip2(iter1, iter2)
  iter1 = seq.iter(iter1)
  iter2 = seq.iter(iter2)
  return function()
    return iter1(), iter2()
  end
end


function seq.zip3(iter1, iter2, iter3)
  iter1 = seq.iter(iter1)
  iter2 = seq.iter(iter2)
  iter3 = seq.iter(iter3)
  return function()
    return iter1(),iter2(),iter3()
  end
end


function get_classifier_options(opt)
  local classifier_opt = {}
  for op, val in pairs(opt) do
    if stringx.startswith(op, 'cl_') then 
      classifier_opt[op:sub(4)] = val
    end
  end
  return classifier_opt
end


function indices_to_string(indices, idx_map)
  local ind, tab = indices, {}
  if torch.type(ind) ~= 'table' then
    ind = ind:totable()
  end
  for _, v in pairs(ind) do 
    if idx_map[v] then
      table.insert(tab, idx_map[v])
    end
  end
  return table.concat(tab, ' ')
end

main()

