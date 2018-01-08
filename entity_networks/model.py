"""
Define the recurrent entity network model.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from functools import partial

import tensorflow as tf

from entity_networks.dynamic_memory_cell import DynamicMemoryCell
from entity_networks.model_ops import cyclic_learning_rate, \
                                      get_sequence_length, \
                                      count_parameters, \
                                      prelu

OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "loss",
    "gradients",
    "gradient_norm",
]

def get_input_encoding_for_gate(inputs, initializer=None, scope=None):
    """
    Implementation of the learned multiplicative mask from Section 2.1, Equation 1.
    This module is also described in [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852)
    as Position Encoding (PE). The mask allows the ordering of words in a sentence to affect the
    encoding.
    """
    with tf.variable_scope(scope, 'Encoding', initializer=initializer):
        _, _, max_sentence_length, embedding_size = inputs.get_shape().as_list()
        positional_mask = tf.get_variable(
            name='positional_mask2',
            shape=[max_sentence_length, embedding_size])

        encoded_input = tf.reduce_sum(inputs * positional_mask, axis=2)
        return encoded_input


def get_input_encoding(inputs, initializer=None, scope=None):
    """
    Implementation of the learned multiplicative mask from Section 2.1, Equation 1.
    This module is also described in [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852)
    as Position Encoding (PE). The mask allows the ordering of words in a sentence to affect the
    encoding.
    """
    with tf.variable_scope(scope, 'Encoding', initializer=initializer):
        _, _, max_sentence_length, embedding_size = inputs.get_shape().as_list()
        positional_mask = tf.get_variable(
            name='positional_mask',
            shape=[max_sentence_length, embedding_size])
        
        encoded_input = tf.reduce_sum(inputs * positional_mask, axis=2)
        return encoded_input

def get_output_module(
        last_state,
        encoded_query,
        num_blocks,
        vocab_size,
        candidates,
        activation=tf.nn.relu,
        initializer=None,
        scope=None):
    """
    Implementation of Section 2.3, Equation 6. This module is also described in more detail here:
    [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852).
    """
    with tf.variable_scope(scope, 'Output', initializer=initializer):
        last_state = tf.stack(tf.split(last_state, num_blocks, axis=1), axis=1)
        _, _, embedding_size = last_state.get_shape().as_list()

        # Use the encoded_query to attend over memories
        # (hidden states of dynamic last_state cell blocks)
        attention = tf.reduce_sum(last_state * encoded_query, axis=2)
	return attention
        # Subtract max for numerical stability (softmax is shift invariant)
        attention_max = tf.reduce_max(attention, axis=-1, keep_dims=True)
        #attention = tf.nn.softmax(attention - attention_max)
        attention = attention - attention_max 
        #with tf.train.SingularMonitoredSession() as sess:
		#sess.run(tf.global_variables_initializer())
	#	sess.run(attention)
        return attention
        #print("attention ", attention)
        #attention = tf.expand_dims(attention, axis=2)

        # Weight memories by attention vectors
        #u = tf.reduce_sum(last_state * attention, axis=1)

        # R acts as the decoder matrix to convert from internal state to the output vocabulary size
        #R = tf.get_variable('R', [embedding_size, vocab_size])
        #H = tf.get_variable('H', [embedding_size, embedding_size])

        #q = tf.squeeze(encoded_query, axis=1)
        #y = tf.matmul(activation(q + tf.matmul(u, H)) , R)
        #Normalize the candidates only and set everything else to 0
        #k_hot = tf.reduce_max(tf.one_hot(candidates, vocab_size),axis=1)
         #print("k-hot ", k_hot)
        #onlyCand = y*k_hot
         #print(onlyCand)
        #summ = tf.reshape(tf.reduce_sum(onlyCand,axis=1),[-1,1])
         #print("summ ", summ)
        #yy = tf.div(onlyCand,summ)
         #print(yy)
         #return yy
        #y = tf.sparse_to_dense(candidates, [tf.shape(candidates, out_type=tf.int64)[0], vocab_size], attention)
        #return y
        batch_s = tf.shape(candidates)[0]
        candsize = candidates.get_shape()[1]
        #phfz = tf.placeholder(tf.float32, shape=[None,vocab_size])
        #zr = tf.Variable(tf.zeros([tf.shape(candidates)[0],vocab_size],dtype=tf.float32))#, validate_shape=False)
        #zr = tf.Variable(tf.tile([[tf.constant(0,dtype=tf.float32)]*vocab_size],[32, 1]))
            #....tf.shape(candidates)[0],1]),validate_shape=False)
        #i = tf.constant(0,dtype=tf.int64)
	#wc = lambda i: tf.less(i, tf.shape(candidates,out_type=tf.int64)[0])

	#b = lambda i: tf.add(i,1) #, tf.concat([_,tf.expand_dims([i]*candsize,0)],0)]
        
          
        #ba = [tf.constant(0,dtype=tf.int64)]
        #print(ba) 
        #arr = ba * candsize

        #tst = [i for i in tf.range(tf.shape(attention)[0])]
        #print(tst)
        #print(arr)
        #earr = tf.expand_dims(arr,0)
        #print(earr)
        #iniv = [i,earr]
        #print(iniv)
        #iniv = [i, tf.expand_dims([tf.cast(0,dtype=tf.int64)]*tf.shape(candidates)[1],0)]

        #si = [i.get_shape(),tf.TensorShape([None,candsize])] #tf.shape(candidates)[1])]
	#r = tf.while_loop(wc, b, [i]) # iniv, shape_invariants=si)
        
        #init = tf.global_variables_initializer()
        #sess = tf.Session()
        #sess.run(init)
        
        #r = sess.run(r)  
        #c = lambda i, _: tf.less(i,r)
        #b = lambda i, _: [tf.add(i, 1), tf.concat([_,tf.expand_dims([i]*candsize,0)],0)]
        #r, indf = tf.while_loop(c, b, iniv, [i.get_shape(),tf.TensorShape([r,candsize])])
        #print("i ", i)
        #print(r)
        #sess = tf.Session()
        #print(sess.run(r))
        #print(sess.run(i))

	fl = tf.reshape(tf.range(tf.shape(candidates,out_type=tf.int64)[0],dtype=tf.int64), [-1,1])
	indf = tf.tile(fl,[1,candsize])
        #print("indf ", indf)
        #ind = tf.concat((tf.Variable([[i]*vocab_size for i in tf.range(tf.shape(candidates)[0])]),candidates),1)
        #ind = tf.concat([indf, candidates], 1)
        ind = tf.reshape(tf.stack([indf, candidates],axis=2),[-1,2])
        #print(ind)
	shape = tf.stack([tf.shape(candidates, out_type=tf.int64)[0],tf.constant(vocab_size, dtype=tf.int64)],axis=0)
        #shape = [tf.shape(candidates, out_type=tf.int64)[0], vocab_size]
        #fixlogit = tf.constant(1100,dtype=tf.float32)
        sparse = tf.SparseTensor(indices=ind, values=tf.reshape(attention,[-1]), dense_shape=shape)
        init = tf.zeros(tf.stack([tf.shape(candidates, out_type=tf.int32)[0],tf.constant(vocab_size, dtype=tf.int32)],axis=0),dtype=tf.float32)
        #init = tf.fill(tf.stack([tf.shape(candidates, out_type=tf.int32)[0],tf.constant(vocab_size, dtype=tf.int32)],axis=0),tf.constant(-1000, dtype=tf.float32))
        
        updated = tf.sparse_add(init, sparse)

        #y = tf.scatter_nd_add(zr, ind, tf.reshape(attention,[-1]))
        #print(updated)
        return updated
    outputs = None
    return outputs

def get_outputs(inputs, params):
    "Return the outputs from the model which will be used in the loss function."
    embedding_size = params['embedding_size']
    num_blocks = params['num_blocks']
    vocab_size = params['vocab_size']

    story = inputs['story']
    query = inputs['query']
    candidates = inputs['candidates']
    print("get outputs candi ", candidates)
    batch_size = tf.shape(story)[0]

    normal_initializer = tf.random_normal_initializer(stddev=0.1)
    ones_initializer = tf.constant_initializer(1.0)

    # Extend the vocab to include keys for the dynamic memory cell,
    # allowing the initialization of the memory to be learned.
    vocab_size = vocab_size + num_blocks

    with tf.variable_scope('EntityNetwork', initializer=normal_initializer):
        # PReLU activations have their alpha parameters initialized to 1
        # so they may be identity before training.
        alpha = tf.get_variable(
            name='alpha',
            shape=embedding_size,
            initializer=ones_initializer)
        activation = partial(prelu, alpha=alpha)

        # Embeddings
        embedding_params = tf.get_variable(
            name='embedding_params',
            shape=[vocab_size, embedding_size])
        #embedding_params = tf.nn.dropout(embedding_params, 0.5)

        # The embedding mask forces the special "pad" embedding to zeros.
        embedding_mask = tf.constant(
            value=[0 if i == 0 else 1 for i in range(vocab_size)],
            shape=[vocab_size, 1],
            dtype=tf.float32)
        embedding_params_masked = embedding_params * embedding_mask
        print("story ", story)
        story_embedding = tf.nn.embedding_lookup(embedding_params_masked, story)
        query_embedding = tf.nn.embedding_lookup(embedding_params_masked, query)
        story_embedding = tf.nn.dropout(story_embedding, 0.2)
        query_embedding = tf.nn.dropout(query_embedding, 0.2)
	print("story embedding ", story_embedding)
        print("query embedding ", query_embedding)
        # Input Module
        encoded_story = get_input_encoding(
            inputs=story_embedding,
            initializer=ones_initializer,
            scope='StoryEncoding')
        encoded_story_for_gate = get_input_encoding_for_gate(
            inputs=story_embedding,
            initializer=ones_initializer,
            scope='StoryEncoding_for_gate')
        encoded_query = get_input_encoding(
            inputs=query_embedding,
            initializer=ones_initializer,
            scope='QueryEncoding')
        
        # Memory Module
        # We define the keys outside of the cell so they may be used for memory initialization.
        # Keys are initialized to a range outside of the main vocab.
        print('model candidates', candidates)
        #keys = [key for key in range(vocab_size - num_blocks, vocab_size)]
        
        keys = tf.nn.embedding_lookup(embedding_params_masked, candidates)
        print("keys ", keys)
        #keys = tf.split(keys, num_blocks, axis=0)
        #print("split keys ", keys)
        #keys = [tf.squeeze(key, axis=0) for key in keys]
        #print("squeezed keys ", keys)
        cell = DynamicMemoryCell(
            num_blocks=num_blocks,
            num_units_per_block=embedding_size,
            keys=keys,
            initializer=normal_initializer,
            recurrent_initializer=normal_initializer,
            activation=activation)
        print(encoded_story)
        print(encoded_story.get_shape())
        inputcat = tf.concat([encoded_story, encoded_story_for_gate], axis=2)
        #inputcat = tf.reshape(inputcat,[tf.shape(encoded_story)[0],encoded_story.get_shape()[1], encoded_story.get_shape()[2]*2])
        print("inputcat ", inputcat)
        # Recurrence
        initial_state = cell.zero_state(batch_size, tf.float32)
        sequence_length = get_sequence_length(encoded_story)
        #assert_op = tf.Assert(tf.equal(0,1), [sequence_length])
        #with tf.control_dependencies([assert_op]):
	#sequence_length = tf.reshape(sequence_length,[-1,1])
	#assertop = tf.Assert(tf.equal(tf.constant(0), tf.constant(1)), data=[tf.reduce_max(sequence_length)],summarize=10)
	#with tf.control_dependencies([assertop]):
	#sequence_length = tf.Print(sequence_length, [sequence_length], message="seq len")
		#with tf.Session() as sess:
			#init = tf.global_variables_initializer()
			#sess.run(init)
			#sess.run(tf.local_variables_initializer())
			#sequence_length.eval(session=sess)
		#	sess.run(tf.constant(5))
		#sequence_length = tf.reshape(sequence_length,[-1,1])
               # sequence_length = sequence_length* tf.constant(5)
	#	sequence_length = tf.reshape(sequence_length,[-1,])
        #print("sequence_length ", sequence_length)
	_, last_state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputcat,
            sequence_length=sequence_length,
            initial_state=initial_state)
 	#print("last state ", last_state)
        # Output Module
        outputs = get_output_module(
            last_state=last_state,
           encoded_query=encoded_query,
            num_blocks=num_blocks,
            vocab_size=vocab_size,
            candidates=candidates,
            initializer=normal_initializer,
            activation=activation)

        parameters = count_parameters()
        print('Parameters: {}'.format(parameters))

        return outputs

def get_predictions(outputs, candidates):
    "Return the actual predictions for use with evaluation metrics or TF Serving."
    
    ### The update here is that, since CBT has 10 candidates, we select them from the vocabsized softmax,
    ### we argmax over the probabilites of the 10 candidates, selecting the index of the highest one.
    ### Then we retreive the original softmax index through the candidates index list.
    ### In these two steps of getting indecies by an index array, we fix the index list to let it match the flattened arrays
    ### then we flatten both arrays, use tf.gather then reshape again
    
    candsize = candidates.shape[1].value
    #candidates = tf.reshape(candidates,[-1,candsize])
  
    ##vocsize = outputs.shape[1]

    #Generate an array to fix the indices to match the flatted array
    ##f = tf.range(tf.shape(outputs, out_type=tf.int64)[0], dtype=tf.int64)*vocsize
    ##f = tf.reshape(f,[-1,1])
    
    #Flatten the fixed candidate array
    ##nidx = tf.reshape(tf.add(f, candidates),[-1])
    
    #Flatten the outputs
    ##foutputs = tf.reshape(outputs,[-1])
    ##onlycandidates = tf.reshape(tf.gather(foutputs,nidx),[-1,candsize])
    
    #Find the index of the highest prob (index in respect to the candidate size)
    predictionsidx = tf.argmax(outputs, axis=-1)
    #assertop = tf.Assert(tf.less_equal(20, 9), [predictionsidx], summarize=10)
    #with tf.control_dependencies([assertop]):
    predictionsidx = tf.reshape(predictionsidx,[-1,1])

    f = tf.range(tf.shape(candidates, out_type=tf.int64)[0], dtype=tf.int64)*candsize
    f = tf.reshape(f,[-1,1])

    nidx = tf.reshape(tf.add(f,predictionsidx), [-1])
    fcandidates = tf.reshape(candidates,[-1])
    predictions = tf.gather(fcandidates,nidx)
    predictions = tf.reshape(predictions,[-1,1])
    #predictions = tf.Print(predictions, [predictions], message="These are predictions: ", first_n=2)
    #with tf.Session() as sess1: 
#	sess1.run(tf.global_variables_initializer())
#	sess1.run(predictions)
     #predictions.eval()
     #predictions = tf.argmax(outputs,axis=-1)
    #assertop = tf.Assert(tf.less_equal(20, 9), [predictions], summarize=10)
    #with tf.control_dependencies([assertop]):
    # 		return predictions
    return predictions

def get_loss(outputs, labels, mode):
    "Return the loss function which will be used with an optimizer."

    loss = None
    if mode == tf.contrib.learn.ModeKeys.INFER:
        return loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=outputs,
        labels=labels)
    return loss

def get_train_op(loss, params, mode):
    "Return the trainining operation which will be used to train the model."

    train_op = None
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
        return train_op

    global_step = tf.contrib.framework.get_or_create_global_step()

    learning_rate = cyclic_learning_rate(
        learning_rate_min=params['learning_rate_min'],
        learning_rate_max=params['learning_rate_max'],
        step_size=params['learning_rate_step_size'],
        mode='triangular',
        global_step=global_step)
    tf.summary.scalar('learning_rate', learning_rate)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer='SGD',
        clip_gradients=params['clip_gradients'],
        gradient_noise_scale=params['gradient_noise_scale'],
        summaries=OPTIMIZER_SUMMARIES)

    return train_op

def model_fn(features, labels, mode, params):
    "Return ModelFnOps for use with Estimator."
    features['candidates'] = tf.reshape(features['candidates'], [-1, 10])    
    candidates = features['candidates']
    #print("model label ", labels)
    #print("model fn", candidates)
    if not labels == None:
	kk =tf.equal(candidates, tf.reshape(labels, [-1,1]))
    	k = tf.where(kk)
    	labels = k[:,1]
    	labels = tf.reshape(labels, [-1])
    
    outputs = get_outputs(features, params)
    
    predictions = get_predictions(outputs, candidates)
     
   # assertop = tf.Assert(tf.less_equal(tf.reduce_max(labels), 9), [labels])
    #with tf.control_dependencies([assertop]):
    loss = get_loss(outputs, labels, mode)
    #print("trainable? ", tf.trainable_variables())
    train_op = get_train_op(loss, params, mode)
    return tf.contrib.learn.ModelFnOps(
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        mode=mode)
