-- uses code from: https://github.com/harvardnlp/seq2seq-attn

local beam = require 's2sa.beam'
require 'nn'
require 'xlua'
require 'optim'
seq = require 'pl.seq'
stringx = require 'pl.stringx'
dbg = require 'debugger'

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
  assert(path.exists(classifier_opt.val_lbl_file), 'val_lbl_file does not exist')
  assert(path.exists(classifier_opt.test_lbl_file), 'test_lbl_file does not exist')
  assert(path.exists(classifier_opt.train_source_file), 'train_source_file does not exist')
  assert(path.exists(classifier_opt.val_source_file), 'val_source_file does not exist')
  assert(path.exists(classifier_opt.test_source_file), 'test_source_file does not exist')
  if classifier_opt.enc_or_dec == 'dec' then
    assert(path.exists(classifier_opt.train_target_file), 'train_target_file does not exist')
    assert(path.exists(classifier_opt.val_target_file), 'val_target_file does not exist')
    assert(path.exists(classifier_opt.test_target_file), 'test_target_file does not exist')    
  end
  assert(path.exists(classifier_opt.save), 'save dir does not exist')
  
  -- number of module for word representation
  module_num = 2*classifier_opt.enc_layer - classifier_opt.use_cell
  -- first pass: get labels
  print('==> first pass: getting labels')
  label2idx, idx2label = get_labels(classifier_opt.train_lbl_file, classifier_opt.semdeprel)
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
  local train_data, val_data, test_data = load_data(classifier_opt, label2idx)

  print('model_opt.brnn: ' .. model_opt.brnn)
  -- use trained encoder/decoder from MT model
  encoder, decoder = model[1], model[2]
  if model_opt.brnn == 1 then
    encoder_brnn = model[4]
  end
  
  -- local word_repr_size
  if classifier_opt.enc_layer > 0 then
    word_repr_size = model_opt.rnn_size
  else
    if model_opt.use_chars_enc == 0 then
      word_repr_size = model_opt.word_vec_size
    else
      word_repr_size = model_opt.num_kernels
    end
    -- TODO handle decoder case with word vectors
  end
      
  local classifier_input_size = word_repr_size
  if classifier_opt.no_dec_repr and (classifier_opt.use_max_attn or classifier_opt.use_min_attn or classifier_opt.use_rand_attn) then
    print('==> not using decoder word representation; instead using only attended word representation')    
    classifier_input_size = 0
  end
  
  if classifier_opt.use_max_attn then
    print('==> using representation of most attended word as additional features')
    -- TODO this assumes encoder/decoder use same representation (hidden, words, char cnn)
    classifier_input_size = classifier_input_size + word_repr_size
  end
  if classifier_opt.use_min_attn then
    print('==> using representation of least attended word as additional features')
    classifier_input_size = classifier_input_size + word_repr_size
  end  
  if classifier_opt.use_rand_attn then
    print('==> using representation of random attended word as additional features')
    classifier_input_size = classifier_input_size + word_repr_size
  end  
  if classifier_opt.use_summary_vec then
    print('==> using summary vector as additional features')
    classifier_input_size = classifier_input_size + model_opt.rnn_size
  end
  if (classifier_opt.deprel or classifier_opt.semdeprel) and classifier_opt.deprel_repr == 'concat' then
    print('==> concatenating head and modifier word representations for predicting dependency relation')
    classifier_input_size = classifier_input_size + word_repr_size
  end
  if classifier_opt.entailment then
    -- This assumes that for entailment classification we only want to concat the sentence representations
    -- and not combine them in any other way
    -- Rocktaschel et. al.(ICLR 2016) and Bowman et. al. (EMNLP 2015) concat sentence representations too
    print('==> concatenating test and hypothesis sentence representations for predicting entailment')
    classifier_input_size = classifier_input_size + classifier_input_size
  end
  
  
  -- define classifier
  classifier = nn.Sequential()
  if classifier_opt.linear_classifier then
    classifier:add(nn.Linear(classifier_input_size, classifier_opt.num_classes))
  else
    classifier:add(nn.Linear(classifier_input_size, classifier_opt.classifier_size))
    classifier:add(nn.Dropout(classifier_opt.classifier_dropout))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Linear(classifier_opt.classifier_size, classifier_opt.num_classes)) 
  end    
  
  --[[
  if classifier_opt.linear_classifier then
    classifier:add(nn.Linear(model_opt.rnn_size, classifier_opt.num_classes))
  else
    if classifier_opt.enc_layer > 0 then
      classifier:add(nn.Linear(model_opt.rnn_size,classifier_opt.classifier_size))
    else
      if model_opt.use_chars_enc == 0 then
        classifier:add(nn.Linear(model_opt.word_vec_size,classifier_opt.classifier_size))    
      else
        classifier:add(nn.Linear(model_opt.num_kernels,classifier_opt.classifier_size))      
      end
      -- TODO handle decoder case with word vectors
    end
    classifier:add(nn.Dropout(classifier_opt.classifier_dropout))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Linear(classifier_opt.classifier_size, classifier_opt.num_classes)) 
  end
  --]]
  
  print('==> defined classification model:')
  print(classifier)
    
  -- define classification criterion
  criterion = nn.CrossEntropyCriterion()
  
  -- move to cuda
  if opt.gpuid >= 0 then     
    classifier = classifier:cuda()
    criterion = criterion:cuda()
  end
  
  -- get classifier parameters and gradients
  classifier_params, classifier_grads = classifier:getParameters()
  
  -- define optimizer
  if classifier_opt.optim == 'ADAM' then
    optim_state = {learningRate = classifier_opt.learning_rate}
    optim_method = optim.adam
  elseif classifier_opt.optim == 'ADAGRAD' then
    optim_state = {learningRate = classifier_opt.learning_rate}
    optim_method = optim.adagrad
  elseif classifier_opt.optim == 'ADADELTA' then
    optim_state = {}
    optim_method = optim.adadelta
  else
    optim_state = {learningRate = classifier_opt.learning_rate}
    optim_method = optim.sgd
  end
  
  
  confusion = optim.ConfusionMatrix(classes)
  
  -- Log results to files
  train_logger = optim.Logger(paths.concat(classifier_opt.save, 'train.log'))
  val_logger = optim.Logger(paths.concat(classifier_opt.save, 'val.log'))  
  test_logger = optim.Logger(paths.concat(classifier_opt.save, 'test.log'), classifier_opt.pred_file)  
  
  collectgarbage(); collectgarbage();
  
  -- do epochs
  local epoch, best_epoch, best_loss = 1, 1, math.huge
  while epoch <= classifier_opt.epochs and epoch - best_epoch <= classifier_opt.patience do 
    if classifier_opt.entailment then
      train_entailment(train_data, epoch)
    else
      train(train_data, epoch)
    end
    local val_loss
    if classifier_opt.entailment then
      val_loss = eval_entailment(val_data, epoch, val_logger, 'val')
    else
      val_loss = eval(val_data, epoch, val_logger, 'val')
    end
    if val_loss < best_loss then
      best_epoch = epoch
      best_loss = val_loss
      if classifier_opt.save_model == 1 then
        -- save current model
        local filename = paths.concat(classifier_opt.save, 'classifier_model_epoch_' .. epoch .. '.t7')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('==> saving model to '..filename)
        torch.save(filename, classifier)        
      end
    end
    if classifier_opt.entailment then
      eval_entailment(test_data, epoch, test_logger, 'test', classifier_opt.pred_file)
    else
      eval(test_data, epoch, test_logger, 'test', classifier_opt.pred_file)
    end
    print('finished epoch ' .. epoch .. ', with val loss: ' .. val_loss)
    print('best epoch: ' .. best_epoch .. ', with val loss: ' .. best_loss)
    epoch = epoch + 1    
    collectgarbage(); collectgarbage();
  end
  if epoch - best_epoch > classifier_opt.patience then
    print('==> reached patience of ' .. classifier_opt.patience .. ' epochs, stopping...')
  end  
end
  
function train(train_data, epoch)
  
  local time = sys.clock()
  classifier:training()
  -- set MT model to evaluate mode
  encoder:evaluate(); decoder:evaluate();
  if model_opt.brnn == 1 then encoder_brnn:evaluate() end
  
  local shuffle = torch.randperm(#train_data)
  
  print('\n==> doing epoch on training data:')
  print('\n==> epoch # ' .. epoch .. ' [batch size = ' .. classifier_opt.batch_size .. ']')
  
  local total_loss, num_total_words = 0, 0
  for i = 1,#train_data, classifier_opt.batch_size do
    collectgarbage()
    xlua.progress(i, #train_data)
    
    -- prepare mini-batch
    local batch_input, batch_labels, batch_heads = {}, {}, {}
    for j = i,math.min(i+classifier_opt.batch_size-1, #train_data) do
      -- TODO: figure out how to edit heads and source
      local source = train_data[shuffle[j]][1]
      if opt.gpuid >= 0 then source = source:cuda() end
      local input, labels, heads = {source}
      if classifier_opt.enc_or_dec == 'enc' then
        if classifier_opt.deprel or classifier_opt.semdeprel then
          heads = train_data[shuffle[j]][2]
          table.insert(batch_heads, heads)
          labels = train_data[shuffle[j]][3]
        elseif classifier_opt.entailment then
          labels = train_data[shuffle[j]][3]
          -- TODO: figure out what to do with the sentences
        else
          labels = train_data[shuffle[j]][2]
        end
      elseif classifier_opt.enc_or_dec == 'dec' then
        local target = train_data[shuffle[j]][2]
        --if opt.gpuid >= 0 then target = target:cuda() end
        target = target:long()
        table.insert(input, target)
        labels = train_data[shuffle[j]][3]
      else
        error('unknown value for classifier_opt.enc_or_dec: ' .. classifier_opt.enc_or_dec)
      end
      table.insert(batch_input, input)
      table.insert(batch_labels, labels)
    end

    -- closure
    local eval_loss_grad = function(x) 
      -- get new params
      if x ~= classifier_params then classifier_params:copy(x) end
      
      -- reset gradients
      classifier_grads:zero()
      
      local loss, num_words = 0, 0
      for j = 1,#batch_input do
        local source = batch_input[j][1]
        if classifier_opt.verbose then 
          print('j: ' .. j)
          print('source:'); print(source);
          print(indices_to_string(source, idx2word_src))
        end
        local source_l = math.min(source:size(1), opt.max_sent_l)
        if classifier_opt.verbose then 
          print('source_l: ' .. source_l)
          print('opt.max_sent_l: ' .. opt.max_sent_l)
        end
        local source_input
        if model_opt.use_chars_enc == 1 then
          source_input = source:view(source_l, 1, source:size(2)):contiguous()
        else
          source_input = source:view(source_l, 1)
        end
        if classifier_opt.verbose then
          print('source_input:'); print(source_input);
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
        if classifier_opt.verbose then print('forward fwd encoder') end
        for t = 1, source_l do
          -- run through encoder if using representations above word vectors or if need it for decoder
          local enc_out
          if classifier_opt.enc_layer > 0 or classifier_opt.enc_or_dec == 'dec' then
            local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
            enc_out = encoder:forward(encoder_input)
            rnn_state_enc = enc_out
            if classifier_opt.verbose then
              print('t: ' .. t)
              print('encoder_input:'); print(encoder_input)
              print('enc_out:'); print(enc_out);
            end
          end
          if classifier_opt.enc_layer > 0 then
            context[{{},t}]:copy(enc_out[module_num])
          else
            local word_vec_out
            if model_opt.use_chars_enc == 0 then              
              if classifier_opt.verbose then print('forwarding word_vecs_enc at t: ' .. t) end
              word_vec_out = word_vecs_enc:forward(source_input[t])
            else
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
          -- forward bwd encoder
          if classifier_opt.verbose then print('forward bwd encoder') end
          for t = source_l, 1, -1 do
            if classifier_opt.enc_layer > 0 or classifier_opt.enc_or_dec == 'dec' then              
              local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
              local enc_out = encoder_brnn:forward(encoder_input)
              rnn_state_enc = enc_out
              context[{{},t}]:add(enc_out[module_num])
              if classifier_opt.verbose then
                print('t: ' .. t)
                print('encoder_input:'); print(encoder_input);
                print('enc_out:'); print(enc_out);
              end
            end
          end
          if model_opt.init_dec == 1 then
            for L = 1, model_opt.num_layers do
              rnn_state_dec[L*2-1+model_opt.input_feed]:add(rnn_state_enc[L*2-1])
              rnn_state_dec[L*2+model_opt.input_feed]:add(rnn_state_enc[L*2])
            end
          end          
        end
        
        local dec_all_out, target_l
        local attn_argmax, attn_argmin, attn_argrand = {}, {}, {}
        if classifier_opt.enc_or_dec == 'dec' then
          local target = batch_input[j][2]
          target_l = math.min(target:size(1), opt.max_sent_l)
          if classifier_opt.verbose then
            print('target:'); print(target);
            print(indices_to_string(target, idx2word_targ))
            print('target_l: ' .. target_l)
          end            
          dec_all_out = context_proto[{{}, {1,target_l}}]:clone() 
          
          -- forward decoder
          if classifier_opt.verbose then print('forward decoder') end
          for t = 2, target_l do 
            local decoder_input1
            if model_opt.use_chars_dec == 1 then
              --decoder_input1 = word2charidx_targ:index(1, target[{{t-1}}]:long())
              decoder_input1 = word2charidx_targ:index(1, target[{{t-1}}])
            else
              decoder_input1 = target[{{t-1}}]
            end
            local decoder_input
            if model_opt.attn == 1 then
              decoder_input = {decoder_input1, context[{{1}}], table.unpack(rnn_state_dec)}
            else
              decoder_input = {decoder_input1, context[{{1}, source_l}], table.unpack(rnn_state_dec)}
            end
            
            -- TODO implement this case
            assert(classifier_opt.enc_layer > 0, 'using word embeddings on decoder side not yet implemented')
              
            local out_decoder = decoder:forward(decoder_input)
            if classifier_opt.use_max_attn or classifier_opt.use_min_attn or classifier_opt.use_rand_attn then
              local out = model[3]:forward(out_decoder[#out_decoder]):float() -- 1 x vocab_size
              local score, index = out:max(2)
              if classifier_opt.use_max_attn then
                local max_attn, max_index = decoder_softmax.output:max(2)
                attn_argmax[t] = max_index[1]
              end
              if classifier_opt.use_min_attn then
                local min_attn, min_index = decoder_softmax.output:min(2)
                attn_argmin[t] = min_index[1]
              end
              if classifier_opt.use_rand_attn then
                local rand_index = torch.random(decoder_softmax.output:size(2))
                attn_argrand[t] = torch.LongTensor{rand_index} -- wrap in tensor for consistency with max/min case
              end            
            end
            
            
            rnn_state_dec = {} -- to be modified later
            if model_opt.input_feed == 1 then
              table.insert(rnn_state_dec, out_decoder[#out_decoder])
            end
            for j = 1, #out_decoder - 1 do
              table.insert(rnn_state_dec, out_decoder[j])
            end
            dec_all_out[{{},t}]:copy(out_decoder[module_num])   
            if classifier_opt.verbose then
              print('t: ' .. t)
              print('decoder_input1:'); print(decoder_input1);
              print('decoder_input:'); print(decoder_input);
              print('out_decoder:'); print(out_decoder);
              print('out:'); print(out);
              print('rnn_state_dec:'); print(rnn_state_dec);
            end
          end                            
        end
        
        -- determine relevant encoder output (happens here b/c may be needed for decoder, when using attn)
        local enc_all_out
        if classifier_opt.use_max_attn or classifier_opt.use_min_attn or classifier_opt.use_rand_attn then
          if not skip_start_end then
            enc_all_out = context
          else
            local end_idx = source_l == opt.max_sent_len and source_l or source_l-1
            enc_all_out = context[{{}, {2,end_idx}}]
          end        
        end
        
        -- take encoder/decoder output as input to classifier
        local classifier_input_all
        if classifier_opt.enc_or_dec == 'dec' then
          -- always ignore start and end sybmols in dec
          local end_idx = target_l == opt.max_sent_len and target_l or target_l-1
          classifier_input_all = dec_all_out[{{}, {2,end_idx}}]
          -- concat summary vector
          if classifier_opt.use_summary_vec then
            local summary_vec =  rnn_state_enc[model_opt.num_layers*2]:view(rnn_state_enc[model_opt.num_layers*2]:nElement())
            classifier_input_all = torch.cat(classifier_input_all[1], torch.expand(summary_vec:view(1, summary_vec:size(1)), classifier_input_all:size(2), summary_vec:size(1)), 2)
            classifier_input_all = classifier_input_all:view(1, classifier_input_all:size(1), classifier_input_all:size(2))
          end
        else
          if not skip_start_end then
            classifier_input_all = context
          else
            local end_idx = source_l == opt.max_sent_len and source_l or source_l-1
            classifier_input_all = context[{{}, {2,end_idx}}]
          end
        end

        if classifier_opt.verbose then 
          print('classifier_input_all:'); print(classifier_input_all);
          print('batch_labels[j]:'); print(batch_labels[j])
          print('string format: ' .. indices_to_string(batch_labels[j], idx2label))
          print('forward/backward classifier')
        end
        -- forward/backward classifier
        for t = 1, classifier_input_all:size(2) do
          local classifier_input = classifier_input_all[{{},t}]
          classifier_input = classifier_input:view(classifier_input:nElement())     
          
          -- options using attention mechanism
          if classifier_opt.use_max_attn then
            local enc_attn_argmax = enc_all_out[{{}, attn_argmax[t+1][1]}]
            classifier_input = torch.cat(classifier_input, enc_attn_argmax:view(enc_attn_argmax:nElement()))
          end
          if classifier_opt.use_min_attn then
            local enc_attn_argmin = enc_all_out[{{}, attn_argmin[t+1][1]}]
            classifier_input = torch.cat(classifier_input, enc_attn_argmin:view(enc_attn_argmin:nElement()))
          end
          if classifier_opt.use_rand_attn then
            local enc_attn_argrand = enc_all_out[{{}, attn_argrand[t+1][1]}]
            classifier_input = torch.cat(classifier_input, enc_attn_argrand:view(enc_attn_argrand:nElement()))
          end
          
          -- TODO: this is a hack because old torch.cat cannot handle empty vectors; simplify after updating
          if classifier_opt.no_dec_repr then
            local offset = word_repr_size
            if classifier_opt.use_summary_vec then offset = offset + model_opt.rnn_size end
            classifier_input = classifier_input[{ {offset+1, classifier_input:size(1)} }]
          end
          
          
          -- semantic dependency relations (may have multiple heads per word so happens separately)
          -- TODO merge this case with the regular case
          if classifier_opt.semdeprel then
            local classifier_input_rel
            for k = 1,#batch_heads[j][t] do 
              if batch_heads[j][t][k] > 0 and batch_labels[j][t][k] > 0 then
                local head_repr = classifier_input_all[{{}, batch_heads[j][t][k]}]
                head_repr = head_repr:view(head_repr:nElement())                
                if classifier_opt.deprel_repr == 'concat' then
                  classifier_input_rel = torch.cat(classifier_input, head_repr)
                else 
                  classifier_input_rel = torch.add(classifier_input, head_repr)
                end
                local classifier_out = classifier:forward(classifier_input_rel)
                loss = loss + criterion:forward(classifier_out, batch_labels[j][t][k])
                num_words = num_words + 1
                local output_grad = criterion:backward(classifier_out, batch_labels[j][t][k])
                classifier:backward(classifier_input_rel, output_grad)
                confusion:add(classifier_out, batch_labels[j][t][k])
              end
            end
            
          -- regular case
          else
          
            -- dependency relations options
            if classifier_opt.deprel and batch_heads[j][t] > 0 then
              --print('batch_heads[j][t]:'); print(batch_heads[j][t])
              local head_repr = classifier_input_all[{{}, batch_heads[j][t]}]
              head_repr = head_repr:view(head_repr:nElement())
              --print('head_repr:'); print(head_repr)
              --print('classifier_input:'); print(classifier_input)
              if classifier_opt.deprel_repr == 'concat' then
                classifier_input = torch.cat(classifier_input, head_repr)
              else 
                classifier_input = torch.add(classifier_input, head_repr)
              end            
              --print('classifier_input:'); print(classifier_input)
            end
            
            if batch_labels[j][t] == 0 then
              print('Warning: skipping word with label idx 0')
            else
              -- don't classify roots
              if not classifier_opt.deprel or batch_heads[j][t] > 0 then
              
                local classifier_out = classifier:forward(classifier_input)
                loss = loss + criterion:forward(classifier_out, batch_labels[j][t])
                num_words = num_words + 1
                local output_grad = criterion:backward(classifier_out, batch_labels[j][t])
                classifier:backward(classifier_input, output_grad)
                
                if classifier_opt.verbose then 
                  print('t: ' .. t)
                  print('classifier_input:'); print(classifier_input);
                  print('classifier_out:'); print(classifier_out);
                  print('batch_labels[j][t]: ' .. batch_labels[j][t])
                  print('loss:'); print(loss);
                  print('output_grad:'); print(output_grad);
                end
                
                -- update confusion matrix
                confusion:add(classifier_out, batch_labels[j][t])
              end
            end
          end
        end    
      end
      
      -- TODO consider normalizing over batch size instead of num words in batch 
      classifier_grads:div(num_words)
      -- keep loss over entire training data
      total_loss = total_loss + loss
      num_total_words = num_total_words + num_words
      -- loss for current batch
      loss = loss/num_words
      
      -- TODO clip gradients?
      
      return loss, classifier_grads      
    end
    
    optim_method(eval_loss_grad, classifier_params, optim_state)
  
  end
  
  time = (sys.clock() - time) / #train_data
  print('==> time to learn 1 sample = ' .. (time*1000) .. 'ms') 
  total_loss = total_loss/num_total_words
  print('==> loss: ' .. total_loss)  
  print(confusion)
  
   -- update logger/plot
  train_logger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  if classifier_opt.plot then
    train_logger:style{['% mean class accuracy (train set)'] = '-'}
    train_logger:plot()
  end  
   
  -- for next epoch
  confusion:zero()
      
end

function train_entailment(train_data, epoch)
  local time = sys.clock()
  classifier:training()
  -- set MT model to evaluate mode
  encoder:evaluate(); decoder:evaluate();
  if model_opt.brnn == 1 then encoder_brnn:evaluate() end

  local shuffle = torch.randperm(#train_data)

  print('\n==> doing epoch on training data:')
  print('\n==> epoch # ' .. epoch .. ' [batch size = ' .. classifier_opt.batch_size .. ']')

  local total_loss, num_total_words = 0, 0
  for i = 1,#train_data, classifier_opt.batch_size do
    collectgarbage()
    xlua.progress(i, #train_data)

    -- prepare mini-batch
    local batch_input, batch_labels = {}, {}
    for j = i,math.min(i+classifier_opt.batch_size-1, #train_data) do
      local t_source = train_data[shuffle[j]][1]
      local h_source = train_data[shuffle[j]][2]
      if opt.gpuid >= 0 then
        t_source = t_source:cuda()
        h_source = h_source:cuda()
      end
      label = train_data[shuffle[j]][3]
      table.insert(batch_input, {t_source, h_source})
      table.insert(batch_labels, label)
    end

    -- closure
    local eval_loss_grad = function(x)
      --  get new params
      if x ~= classifier_params then classifier_params:copy(x) end

      -- reset gradients
      classifier_grads:zero()

      local loss, num_words = 0, 0
      for j = 1,#batch_input do
        local t_source, h_source = batch_input[j][1], batch_input[j][2]
        if classifier_opt.verbose then
          print('j: ' .. j)
          print('t_source:'); print(t_source);
          print('t_sent: ' .. indices_to_string(t_source, idx2word_src))
          print('h_source:'); print(h_source);
          print('h_sent: ' .. indices_to_string(h_source, idx2word_src))
        end
        local t_source_l, h_source_l = math.min(t_source:size(1), opt.max_sent_l), math.min(h_source:size(1), opt.max_sent_l)
        if classifier_opt.verbose then
          print('t_source_l: ' .. t_source_l)
          print('h_source_l: ' .. h_source_l)
          print('opt.max_sent_l: ' .. opt.max_sent_l)
        end
        local t_source_input, h_source_input
        if model_opt.use_chars_enc == 1 then
          t_source_input = t_source:view(t_source_l, 1, t_source:size(2)):contiguous()
          h_source_input = h_source:view(h_source_l, 1, h_source:size(2)):contiguous()
        else
          t_source_input = t_source:view(t_source_l, 1)
          h_source_input = h_source:view(h_source_l, 1)
        end
        if classifier_opt.verbose then
          print('t_source_input:'); print(t_source_input);
          print('h_source_input:'); print(h_source_input);
        end

        local rnn_state_enc = {}
        for i = 1, #init_fwd_enc do
          table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
        end
        local t_context = context_proto[{{}, {1,t_source_l}}]:clone() -- 1 x source_l x rnn_size
        local h_context = context_proto[{{}, {1,h_source_l}}]:clone() -- 1 x source_l x rnn_size
        -- special case when using word vectors
        if classifier_opt.enc_layer == 0 then
          if model_opt.use_chars_enc == 0 then
            t_context = context_proto_word_vecs[{ {}, {1,t_source_l}, {} }]:clone()
            h_context = context_proto_word_vecs[{ {}, {1,h_source_l}, {} }]:clone()
          else
            t_context = context_proto_char_cnn[{ {}, {1,t_source_l}, {} }]:clone()
            h_context = context_proto_char_cnn[{ {}, {1,h_source_l}, {} }]:clone()
          end
        end

        -- forward encoder
        if classifier_opt.verbose then print('forward fwd encoder') end
        -- run teext sentence through the encoder
        local final_t_enc_out, final_h_enc_out
        for t = 1, t_source_l do
          local t_enc_out
          if classifier_opt.enc_layer > 0 then
            local t_encoder_input = {t_source_input[t], table.unpack(rnn_state_enc)}
            t_enc_out = encoder:forward(t_encoder_input)
            rnn_state_enc = t_enc_out
            if classifier_opt.verbose then
              print('t_encoder_input:'); print(t_encoder_input)
              print('t_enc_out:'); print(t_enc_out);
            end
          end
          final_t_enc_out = t_enc_out
          if classifier_opt.enc_layer > 0 then
            t_context[{{},t}]:copy(t_enc_out[module_num])
          end
        end

        -- run premise sentence through brnn
        if model_opt.brnn == 1 then
          for i = 1, #rnn_state_enc do
            rnn_state_enc[i]:zero()
          end
          -- forward bwd encoder
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

        -- run hypothesis sentence through the encoder
        -- first refresh the rnn_state_encoder
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
            if classifier_opt.verbose then
              print('encoder_input:'); print(h_encoder_input)
              print('enc_out:'); print(h_enc_out);
            end
          end
          final_h_enc_out = h_enc_out
          if classifier_opt.enc_layer > 0 then
            h_context[{{},t}]:copy(h_enc_out[module_num])
          end
        end

        -- run hypothesis sentence through brnn
        if model_opt.brnn == 1 then
          for i = 1, #rnn_state_enc do
            rnn_state_enc[i]:zero()
          end
          -- forward bwd encoder
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


        -- combine encoded t and h sentences for the classifier
        local classifier_input
        if model_opt.brnn == 1 then
          classifier_input = torch.cat(t_context[{1,1}], h_context[{1,1}])
        else
          classifier_input = torch.cat(t_context[{1,t_source_l}], h_context[{1,h_source_l}])
        end
        -- take encoder output as input to classifier
        local classifier_out = classifier:forward(classifier_input)
        loss = loss + criterion:forward(classifier_out, batch_labels[j])
        num_words = num_words + 1
        local output_grad = criterion:backward(classifier_out, batch_labels[j])
        classifier:backward(classifier_input, output_grad)

        if classifier_opt.verbose then
          print('j: ' .. j)
          print('classifier_input:'); print(classifier_input);
          print('classifier_out:'); print(classifier_out);
          print('batch_labels[j]: ' .. batch_labels[j])
          print('loss:'); print(loss);
          print('output_grad:'); print(output_grad);
          end

        -- update confusion matrix
        confusion:add(classifier_out, batch_labels[j])
      end

      classifier_grads:div(num_words)
      -- keep loss over entire training data
      total_loss = total_loss + loss
      num_total_words = num_total_words + num_words
      -- loss for current batch
      loss = loss/num_words

      return loss, classifier_grads
    end

    optim_method(eval_loss_grad, classifier_params, optim_stat)
  end

  time = (sys.clock() - time) / #train_data
  print('==> time to learn 1 sample = ' .. (time*1000) .. 'ms')
  total_loss = total_loss/num_total_words
  print('==> loss: ' .. total_loss)
  print(confusion)

  -- update logger/plot
  train_logger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  if classifier_opt.plot then
    train_logger:style{['% mean class accuracy (train set)'] = '-'}
    train_logger:plot()
  end

  -- for next epoch
  confusion:zero()

end

function eval(data, epoch, logger, test_or_val, pred_filename)
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
    local source, target, labels, heads = data[i][1]
    if opt.gpuid >= 0 then source = source:cuda() end
    if classifier_opt.enc_or_dec == 'enc' then
      if classifier_opt.deprel or classifier_opt.semdeprel then
        heads = data[i][2]
        labels = data[i][3]
      else
        labels = data[i][2]      
      end
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
        if classifier_opt.enc_layer > 0 or classifier_opt.enc_or_dec == 'dec' then
          local encoder_input = {source_input[t], table.unpack(rnn_state_enc)}
          local enc_out = encoder_brnn:forward(encoder_input)
          rnn_state_enc = enc_out
          context[{{},t}]:add(enc_out[module_num])
        end
      end
      if model_opt.init_dec == 1 then
        for L = 1, model_opt.num_layers do
          rnn_state_dec[L*2-1+model_opt.input_feed]:add(rnn_state_enc[L*2-1])
          rnn_state_dec[L*2+model_opt.input_feed]:add(rnn_state_enc[L*2])
        end
      end                
    end
    
    local dec_all_out, target_l
    local attn_argmax, attn_argmin, attn_argrand = {}, {}, {}
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
        if classifier_opt.use_max_attn or classifier_opt.use_min_attn or classifier_opt.use_rand_attn then
          local out = model[3]:forward(out_decoder[#out_decoder]):float() -- 1 x vocab_size
          local score, index = out:max(2)
          if classifier_opt.use_max_attn then
            local max_attn, max_index = decoder_softmax.output:max(2)
            attn_argmax[t] = max_index[1]
          end
          if classifier_opt.use_min_attn then
            local mix_attn, min_index = decoder_softmax.output:min(2)
            attn_argmin[t] = min_index[1]
          end
          if classifier_opt.use_rand_attn then
            local rand_index = torch.random(decoder_softmax.output:size(2))
            attn_argrand[t] = torch.LongTensor{rand_index} -- wrap in tensor for consistency with max/min case
          end
        end
        
        
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
    
    
    -- determine relevant encoder output (happens here b/c may be needed for decoder, when using attn)
    local enc_all_out
    if classifier_opt.use_max_attn or classifier_opt.use_min_attn or classifier_opt.use_rand_attn then
      if not skip_start_end then
        enc_all_out = context
      else
        local end_idx = source_l == opt.max_sent_len and source_l or source_l-1
        enc_all_out = context[{{}, {2,end_idx}}]
      end
    end
    
    -- take encoder/decoder output as input to classifier
    local classifier_input_all
    if classifier_opt.enc_or_dec == 'dec' then
      -- always ignore start and end sybmols in dec
      local end_idx = target_l == opt.max_sent_len and target_l or target_l-1
      classifier_input_all = dec_all_out[{{}, {2,end_idx}}]
      -- concat summary vector
      if classifier_opt.use_summary_vec then
        local summary_vec =  rnn_state_enc[model_opt.num_layers*2]:view(rnn_state_enc[model_opt.num_layers*2]:nElement())
        classifier_input_all = torch.cat(classifier_input_all[1], torch.expand(summary_vec:view(1, summary_vec:size(1)), classifier_input_all:size(2), summary_vec:size(1)), 2)
        classifier_input_all = classifier_input_all:view(1, classifier_input_all:size(1), classifier_input_all:size(2))
      end      
    else
      if not skip_start_end then
        classifier_input_all = context
      else
        local end_idx = source_l == opt.max_sent_len and source_l or source_l-1
        classifier_input_all = context[{{}, {2,end_idx}}]
      end
    end

    -- write out representations if needed
    if classifier_opt.write_test_word_repr and word_repr_file then
      for t = 1, classifier_input_all:size(2) do
        word_counter = word_counter + 1
        local word_repr = classifier_input_all[{{}, t}]
        word_repr = word_repr:view(word_repr:nElement()) 
        word_repr_file:writeString(tostring(word_counter) .. ' ')
        word_repr_file:writeDouble(word_repr:double():storage())        
      end
    end
    
    -- forward classifier    
    local pred_labels = {}
    for t = 1, classifier_input_all:size(2) do
      -- take word representation
      local classifier_input = classifier_input_all[{{},t}]      
      classifier_input = classifier_input:view(classifier_input:nElement())     
      
      if classifier_opt.use_max_attn then
        local enc_attn_argmax = enc_all_out[{{}, attn_argmax[t+1][1]}]
        classifier_input = torch.cat(classifier_input, enc_attn_argmax:view(enc_attn_argmax:nElement()))
      end
      if classifier_opt.use_min_attn then
        local enc_attn_argmin = enc_all_out[{{}, attn_argmin[t+1][1]}]
        classifier_input = torch.cat(classifier_input, enc_attn_argmin:view(enc_attn_argmin:nElement()))
      end
      if classifier_opt.use_rand_attn then
        local enc_attn_argrand = enc_all_out[{{}, attn_argrand[t+1][1]}]
        classifier_input = torch.cat(classifier_input, enc_attn_argrand:view(enc_attn_argrand:nElement()))
      end
      
      -- TODO: this is a hack because old torch.cat cannot handle empty vectors; simplify after updating
      if classifier_opt.no_dec_repr then
        local offset = word_repr_size
        if classifier_opt.use_summary_vec then offset = offset + model_opt.rnn_size end
        classifier_input = classifier_input[{ {offset+1, classifier_input:size(1)} }]
      end
      
            
      -- semantic dependency relations (may have multiple heads per word so happens separately)
      -- TODO merge this case with the regular case
      if classifier_opt.semdeprel then
        local cur_pred_labels = {}
        local classifier_input_rel
        for k = 1,#heads[t] do           
          if heads[t][k] > 0 and labels[t][k] > 0 then
            local head_repr = classifier_input_all[{{}, heads[t][k]}]
            head_repr = head_repr:view(head_repr:nElement())            
            if classifier_opt.deprel_repr == 'concat' then
              classifier_input_rel = torch.cat(classifier_input, head_repr)
            else 
              classifier_input_rel = torch.add(classifier_input, head_repr)
            end
            
            local classifier_out = classifier:forward(classifier_input_rel)
            loss = loss + criterion:forward(classifier_out, labels[t][k])
            
            if pred_file then
              local _, pred_idx =  classifier_out:max(1)
              pred_idx = pred_idx:long()[1]
              local pred_label = idx2label[pred_idx]
              table.insert(cur_pred_labels, pred_label)
            end
            
            num_words = num_words + 1
            confusion:add(classifier_out, labels[t][k])
            
          else
            if pred_file then
              -- insert null label
              table.insert(cur_pred_labels, '_')
            end
          end
        end
        
        if pred_file then
          -- combine multiple heads
          table.insert(pred_labels, stringx.join('|', cur_pred_labels))
        end
        
      -- regular case
      else
      
        -- dependency relations options
        if classifier_opt.deprel and heads[t] > 0 then
          local head_repr = classifier_input_all[{{}, heads[t]}]
          head_repr = head_repr:view(head_repr:nElement())
          if classifier_opt.deprel_repr == 'concat' then
            classifier_input = torch.cat(classifier_input, head_repr)
          else 
            classifier_input = torch.add(classifier_input, head_repr)
          end            
        end
        
        if labels[t] == 0 then
          print('Warning: skipping word with label idx 0')
        else
          -- don't classify roots
          if not classifier_opt.deprel or heads[t] > 0 then
                            
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
        end
      end
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
  if word_repr_file then word_repr_file:close() end
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

    --TODO: need to figure out how to deal with the situation where opt.max_sent_l < _source:size(1)
    -- Probably best move is to just do the equivalent to continue in python
    -- https://stackoverflow.com/questions/3524970/why-does-lua-have-no-continue-statement
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
    -- special case when using word vectors
    if classifier_opt.enc_layer == 0 then
      if model_opt.use_chars_enc == 0 then
        t_context = context_proto_word_vecs[{ {}, {1,t_source_l}, {} }]:clone()
        h_context = context_proto_word_vecs[{ {}, {1,h_source_l}, {} }]:clone()
      else
        t_context = context_proto_char_cnn[{ {}, {1,t_source_l}, {} }]:clone()
        h_context = context_proto_char_cnn[{ {}, {1,h_source_l}, {} }]:clone()
      end
    end

    --forward encoder for t sentence
    local rnn_state_enc = {}
    for i = 1, #init_fwd_enc do
      table.insert(rnn_state_enc, init_fwd_enc[i]:zero())
    end
    local pred_labels = {}
    for t = 1, t_source_l do
      -- run through encoder if using representations above word vectors or if need it for decoder
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

    -- run premise sentence through brnn
    if model_opt.brnn == 1 then
      for i = 1, #rnn_state_enc do
        rnn_state_enc[i]:zero()
      end
      -- forward bwd encoder
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
      -- run through encoder if using representations above word vectors or if need it for decoder
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

    -- run premise sentence through brnn
    if model_opt.brnn == 1 then
      for i = 1, #rnn_state_enc do
        rnn_state_enc[i]:zero()
      end
      -- forward bwd encoder
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

    -- combine encoded t and h sentences for the classifier
    local classifier_input
    if model_opt.brnn == 1 then
      classifier_input = torch.cat(t_context[{1,1}], h_context[{1,1}])
    else
      classifier_input = torch.cat(t_context[{1,t_source_l}], h_context[{1,h_source_l}])
    end
    local classifier_out = classifier:forward(classifier_input)
    -- get predicted labels to write to file
    if pred_file then
      local _, pred_idx =  classifier_out:max(1)
      pred_idx = pred_idx:long()[1]
      local pred_label = idx2label[pred_idx]
      table.insert(pred_labels, pred_label)
    end

    loss = loss + criterion:forward(classifier_out, label)
    num_words = num_words + 1

    confusion:add(classifier_out, label)

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
  if word_repr_file then word_repr_file:close() end
  return loss
end

function load_data(classifier_opt, label2idx)
  local train_data, val_data, test_data
  if classifier_opt.enc_or_dec == 'enc' then
    if classifier_opt.deprel or classifier_opt.semdeprel then 
      unknown_labels = 0
      train_data = load_source_head_data(classifier_opt.train_source_file, classifier_opt.train_head_file, classifier_opt.train_lbl_file, label2idx, classifier_opt.max_sent_len) 
      print('==> words with unknown labels in train data: ' .. unknown_labels)
      unknown_labels = 0
      val_data = load_source_head_data(classifier_opt.val_source_file, classifier_opt.val_head_file, classifier_opt.val_lbl_file, label2idx) 
      print('==> words with unknown labels in val data: ' .. unknown_labels)
      unknown_labels = 0
      test_data = load_source_head_data(classifier_opt.test_source_file, classifier_opt.test_head_file, classifier_opt.test_lbl_file, label2idx)   
      print('==> words with unknown labels in test data: ' .. unknown_labels)      
    elseif classifier_opt.entailment then
      -- if any of these unknown_labels are 0 then there is an issue
      unknown_labels = 0
      train_data = load_source_entailment_data(classifier_opt.train_source_file, classifier_opt.train_orig_dataset_file, classifier_opt.train_lbl_file, label2idx, classifier_opt.max_sent_len)
      print('==> words with unknown labels in train data: ' .. unknown_labels)
      assert(unknown_labels == 0, 'Training data contained an unknown label')
      unknown_labels = 0
      val_data = load_source_entailment_data(classifier_opt.val_source_file, classifier_opt.val_orig_dataset_file, classifier_opt.val_lbl_file, label2idx)
      print('==> words with unknown labels in val data: ' .. unknown_labels)
      assert(unknown_labels == 0, 'Validation data contained an unknown label')
      unknown_labels = 0
      test_data = load_source_entailment_data(classifier_opt.test_source_file, classifier_opt.test_orig_dataset_file, classifier_opt.test_lbl_file, label2idx)
      print('==> words with unknown labels in test data: ' .. unknown_labels)
      assert(unknown_labels == 0, 'Test data contained an unknown label')
    else
      unknown_labels = 0
      train_data = load_source_data(classifier_opt.train_source_file, classifier_opt.train_lbl_file, label2idx, classifier_opt.max_sent_len)
      print('==> words with unknown labels in train data: ' .. unknown_labels)
      unknown_labels = 0
      val_data = load_source_data(classifier_opt.val_source_file, classifier_opt.val_lbl_file, label2idx)
      print('==> words with unknown labels in val data: ' .. unknown_labels)
      unknown_labels = 0
      test_data = load_source_data(classifier_opt.test_source_file, classifier_opt.test_lbl_file, label2idx)
      print('==> words with unknown labels in test data: ' .. unknown_labels)
    end
  else
    unknown_labels = 0
    train_data = load_source_target_data(classifier_opt.train_source_file, classifier_opt.train_target_file, classifier_opt.train_lbl_file, label2idx, classifier_opt.max_sent_len) 
    print('==> words with unknown labels in train data: ' .. unknown_labels)
    unknown_labels = 0
    val_data = load_source_target_data(classifier_opt.val_source_file, classifier_opt.val_target_file, classifier_opt.val_lbl_file, label2idx) 
    print('==> words with unknown labels in val data: ' .. unknown_labels)
    unknown_labels = 0
    test_data = load_source_target_data(classifier_opt.test_source_file, classifier_opt.test_target_file, classifier_opt.test_lbl_file, label2idx)   
    print('==> words with unknown labels in test data: ' .. unknown_labels)    
  end
  return train_data, val_data, test_data
end


function load_source_data(file, label_file, label2idx, max_sent_len)
  local max_sent_len = max_sent_len or math.huge
  local data = {}
  for line, labels in seq.zip(io.lines(file), io.lines(label_file)) do
    local source
    sent = beam.clean_sent(line)
    if model_opt.use_chars_enc == 0 then
      source, _ = beam.sent2wordidx(line, word2idx_src, model_opt.start_symbol)
    else
      source, _ = beam.sent2charidx(line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
    end
    local label_idx, idx = {}
    for label in labels:gmatch'([^%s]+)' do
      if label2idx[label] then
        idx = label2idx[label]
      else
        print('Warning: using idx 0 for unknown label ' .. label .. ' in line: ' .. line .. ' with labels: ' .. labels)
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
    
    local label_idx, idx = {}
    for label in labels:gmatch'([^%s]+)' do
      if label2idx[label] then
        idx = label2idx[label]
      else
        print('Warning: using idx 0 for unknown label ' .. label .. ' in target line: ' .. target_line .. ' with labels: ' .. labels)
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

-- load source words and their head indices
function load_source_head_data(file, head_file, label_file, label2idx, max_sent_len) 
  local max_sent_len = max_sent_len or math.huge
  local data = {}
  for line, head_line, labels in seq.zip3(io.lines(file), io.lines(head_file), io.lines(label_file)) do
    --print(line)
    --print(head_line)
    --print(labels)
    --print('------------------------')
    --sent = beam.clean_sent(line)
    local source
    if model_opt.use_chars_enc == 0 then
      source, _ = beam.sent2wordidx(line, word2idx_src, model_opt.start_symbol)
    else
      source, _ = beam.sent2charidx(line, char2idx, model_opt.max_word_l, model_opt.start_symbol)
    end
    if source:dim() == 0 then
      print('Warning: empty source vector in line ' .. line)
    end
    local label_idx, idx = {}
    for label in labels:gmatch'([^%s]+)' do
      if classifier_opt.semdeprel then 
        idx = {}
        for _, l in pairs(stringx.split(label, '|')) do
          if label2idx[l] then
            table.insert(idx, label2idx[l])
          else
            print('Warning: using idx 0 for unknown label ' .. label .. ' in line: ' .. line .. ' with labels: ' .. labels)
            table.insert(idx, 0)
            unknown_labels = unknown_labels + 1
          end
        end        
      else
        if label2idx[label] then
          idx = label2idx[label]
        else
          print('Warning: using idx 0 for unknown label ' .. label .. ' in line: ' .. line .. ' with labels: ' .. labels)
          idx = 0
          unknown_labels = unknown_labels + 1
        end
      end
      table.insert(label_idx, idx)
    end
    local heads = {}
    for head in head_line:gmatch'([^%s]+)' do
      if classifier_opt.semdeprel then
        local multheads = {}
        for _, h in pairs(stringx.split(head, '|')) do
          table.insert(multheads, tonumber(h))
        end
        table.insert(heads, multheads)
      else
        table.insert(heads, tonumber(head))
      end
    end
    if #label_idx <= max_sent_len then
      table.insert(data, {source, heads, label_idx})
    end
  end
  return data
end

-- load pair of sentences for entailment classification
function load_source_entailment_data(file, dataset_file, label_file, label2idx, max_sent_len)
  local max_sent_len = max_sent_len or math.huge
  data = {}
  for sents, orig_source, label in seq.zip3(io.lines(file), io.lines(dataset_file), io.lines(label_file)) do
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
    h_l, t_l = #stringx.split(h_sent, " "), #stringx.split(t_sent, " ")
    if h_l <= max_sent_len and h_l >= 1 and t_l <= max_sent_len and t_l >= 1 then
      table.insert(data, {t_source, h_source, label2idx[label]})
    end
  end
  return data
end

function get_labels(label_file, multilabel)
  local label2idx, idx2label = {}, {}
  for line in io.lines(label_file) do
    for label in line:gmatch'([^%s]+)' do
      -- if word has more than one label, separated by "|"
      if multilabel then
        for _, l in pairs(stringx.split(label, '|')) do
          if not label2idx[l] then
            idx2label[#idx2label+1] = l
            label2idx[l] = #idx2label
          end
        end
      else
        if not label2idx[label] then
          idx2label[#idx2label+1] = label
          label2idx[label] = #idx2label
        end
      end
    end
  end
  return label2idx, idx2label
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

