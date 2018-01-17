import tensorflow as tf

class ProjectionOp:
    """ Single layer perceptron
    Project input tensor on the output dimension
    """
    def __init__(self, shape, scope=None, dtype=None):
        """
        Args:
            shape: a tuple (input dim, output dim)
            scope (str): encapsulate variables
            dtype: the weights type
        """
        assert len(shape) == 2

        self.scope = scope

        # Projection on the keyboard
        with tf.variable_scope('weights_' + self.scope):
            self.W_t = tf.get_variable(
                'weights',
                shape,
                # initializer=tf.truncated_normal_initializer()  # TODO: Tune value (fct of input size: 1/sqrt(input_dim))
                dtype=dtype
            )
            self.b = tf.get_variable(
                'bias',
                shape[0],
                initializer=tf.constant_initializer(),
                dtype=dtype
            )
            self.W = tf.transpose(self.W_t)

    def getWeights(self):
        """ Convenience method for some tf arguments
        """
        return self.W, self.b

    def __call__(self, X):
        """ Project the output of the decoder into the vocabulary space
        Args:
            X (tf.Tensor): input value
        """
        with tf.name_scope(self.scope):
            return tf.matmul(X, self.W) + self.b


class seq2seq:
    def __init__(self, args, text_data):

        self.args = args
        self.text_data = text_data

        # Placeholders
        self.encoder_inputs = None
        self.decoder_inputs = None
        self.decoder_targets = None
        self.decoder_weights = None

        self.num_encoder_symbols = len(text_data.sr_word2id)
        self.num_decoder_symbols = self.num_encoder_symbols

        # self.num_encoder_symbols = 10000
        # self.num_decoder_symbols = 10000

        # important operation
        self.outputs = None
        self.loss = None

        self.build_model()

    def build_model(self):

        outputProjection = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < self.args.softmaxSamples < self.text_data.getVocabularySize():
            outputProjection = ProjectionOp(
                (self.text_data.getVocabularySize(), self.args.hiddenSize),
                scope='softmax_projection',
                dtype=tf.float32
            )

            def sampledSoftmax(labels, inputs):
                labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                localWt = tf.cast(outputProjection.W_t, tf.float32)
                localB = tf.cast(outputProjection.b, tf.float32)
                localInputs = tf.cast(inputs, tf.float32)

                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        localWt,  # Should have shape [num_classes, dim]
                        localB,
                        labels,
                        localInputs,
                        self.args.softmaxSamples,  # The number of classes to randomly sample per batch
                        self.text_data.getVocabularySize()),  # The number of classes
                    tf.float32)

        # define mutil RNN cell
        def create_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(self.args.hidden_size)
            cell = tf.contrib.rnn.DropoutWrapper(
                        cell,
                        input_keep_prob=1.0,
                        output_keep_prob=self.args.dropout)
            return cell

        self.cell = tf.contrib.rnn.MultiRNNCell([create_cell() for _ in range(self.args.rnn_layers)])

        # define placeholder
        with tf.name_scope("encoder_placeholder"):
            self.encoder_inputs = [tf.placeholder(tf.int32, [None, ])
                                    for _ in range(self.args.maxLengthEnco)]
        with tf.name_scope("decoder_placeholder"):
            self.decoder_inputs  = [tf.placeholder(tf.int32,   [None, ], name='decoder_inputs')
                                    for _ in range(self.args.maxLengthDeco)]
            self.decoder_targets  = [tf.placeholder(tf.int32,   [None, ], name='decoder_targets')
                                    for _ in range(self.args.maxLengthDeco)]
            self.decoder_weights  = [tf.placeholder(tf.float32,   [None, ], name='decoder_weights')
                                    for _ in range(self.args.maxLengthDeco)]



        decoder_output, state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.encoder_inputs,
                           self.decoder_inputs,
                           self.cell,
                           self.num_encoder_symbols,
                           self.num_decoder_symbols,
                           self.args.embedding_size,
                           output_projection=None,
                           feed_previous=bool(self.args.test),
                           dtype=None,
                           scope=None)

        # For testing only
        if self.args.test is not None:
            if not outputProjection:
                self.outputs = decoder_output
            else:
                self.outputs = [outputProjection(output) for output in decoder_output]
        else:
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(logits=decoder_output,
                                               targets=self.decoder_targets,
                                               weights=self.decoder_weights)
            tf.summary.scalar('loss', self.loss)  # Keep track of the cost


        print("模型构建完毕")

    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        feedDict = {}

        if not self.args.test:  # Training
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoder_inputs[i]]  = batch.encoderSeqs[i]
            for i in range(self.args.maxLengthDeco):
                feedDict[self.decoder_inputs[i]]  = batch.decoderSeqs[i]
                feedDict[self.decoder_targets[i]] = batch.targetSeqs[i]
                feedDict[self.decoder_weights[i]] = batch.weights[i]

            # ops = (self.optOp, self.lossFct)
        else:  # Testing (batchSize == 1)
            for i in range(self.args.maxLengthEnco):
                feedDict[self.encoder_inputs[i]]  = batch.encoderSeqs[i]
            feedDict[self.decoder_inputs[0]]  = [self.text_data.sr_word2id[self.text_data.goToken]]

            # ops = (self.outputs,)

        # Return one pass operator
        return feedDict



