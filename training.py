from data_utils.cornell.data_util import TextData
from seq2seq_model.seq2seq_model import seq2seq
from args import args
import tensorflow as tf
import datetime
import os
from tqdm import tqdm
import math


class ChatBot:
    def __init__(self):

        self.args = args()

        self.text_data = None

        self.global_step = 0

        self.SENTENCES_PREFIX = ['Q: ', 'A: ']

        self.main()

    def main(self):

        self.text_data = TextData(self.args)

        with tf.Graph().as_default():

            # build seq2seq model
            self.seq2seq_model = seq2seq(self.args, self.text_data)

            # Saver/summaries
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "save/model", self.args.corpus_name))
            self.writer = tf.summary.FileWriter(out_dir)
            self.saver = tf.train.Saver()

            session_conf = tf.ConfigProto(
                allow_soft_placement=self.args.allow_soft_placement,
                log_device_placement=self.args.log_device_placement)
            self.sess = tf.Session(config=session_conf)

            if self.args.test is not None:
                self.managePreviousModel()

            if self.args.test == 'interactive':
                self.main_test_interactive()
            elif self.args.test == 'web_interface':
                # TODO: web interface
                self.main_test_web()
            else:
                self.main_train()
                
    
    def main_train(self):

        mergedSummaries = tf.summary.merge_all()

        optimizer = tf.train.AdamOptimizer(self.args.learning_rate,
                                           beta1=self.args.beta1,
                                           beta2=self.args.beta2,
                                           epsilon=self.args.epsilon
                                           )

        grads_and_vars = optimizer.compute_gradients(self.seq2seq_model.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

        self.sess.run(tf.global_variables_initializer())

        try:  # If the user exit while training, we still try to save the model
            for i in range(self.args.epoch_nums):

                # Generate batches
                tic = datetime.datetime.now()
                batches = self.text_data.get_next_batches()
                for next_batch in tqdm(batches, desc="Training"):
                    # step, summaries, loss = self.seq2seq_model.step(next_batch)
                    feed_dict = self.seq2seq_model.step(next_batch)

                    _, summaries, loss = self.sess.run(
                        (self.train_op, mergedSummaries, self.seq2seq_model.loss),
                        feed_dict)
                    self.global_step += 1

                    self.writer.add_summary(summaries, self.global_step)

                    # Output training status
                    if self.global_step % 100 == 0:
                        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                        tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" %(self.global_step, loss, perplexity))

                    if self.global_step % self.args.checkpoint_every == 0:
                        self.save_session(self.sess, self.global_step)

                toc = datetime.datetime.now()
                print("Epoch finished in {}".format(toc - tic))

        except (KeyboardInterrupt, SystemExit):  # If the user press Ctrl+C while testing progress
            print('Interruption detected, exiting the program...')

        # self.save_session(sess, self.global_step)  # Ultimate saving before complete exit


    def main_test_interactive(self):
        """ Try predicting the sentences that the user will enter in the console
        Args:
            sess: The current running session
        """
        # TODO: If verbose mode, also show similar sentences from the training set with the same words (include in mainTest also)
        # TODO: Also show the top 10 most likely predictions for each predicted output (when verbose mode)
        # TODO: Log the questions asked for latter re-use (merge with test/samples.txt)

        print('Testing: Launch interactive mode:')
        print('')
        print('Welcome to the interactive mode, here you can ask to Deep Q&A the sentence you want. Don\'t have high '
              'expectation. Type \'exit\' or just press ENTER to quit the program. Have fun.')

        # Initialize all variables
        # self.sess.run(tf.global_variables_initializer())

        while True:
            question = input(self.SENTENCES_PREFIX[0])
            if question == '' or question == 'exit':
                break

            questionSeq = []  # Will be contain the question as seen by the encoder
            answer = self.singlePredict(question, questionSeq)
            if not answer:
                print('Warning: sentence too long, sorry. Maybe try a simpler sentence.')
                continue  # Back to the beginning, try again

            print('{}{}'.format(self.SENTENCES_PREFIX[1], self.text_data.sequence2str(answer, clean=True)))

            print()

    def singlePredict(self, question, questionSeq=None):
        """ Predict the sentence
        Args:
            question (str): the raw input sentence
            questionSeq (List<int>): output argument. If given will contain the input batch sequence
        Return:
            list <int>: the word ids corresponding to the answer
        """
        # Create the input batch
        batch = self.text_data.sentence2enco(question)
        if not batch:
            return None
        if questionSeq is not None:  # If the caller want to have the real input
            questionSeq.extend(batch.encoderSeqs)

        # Run the model
        feed_dict = self.seq2seq_model.step(batch)

        output = self.sess.run(self.seq2seq_model.outputs, feed_dict)  # TODO: Summarize the output too (histogram, ...)
        answer = self.text_data.deco2sentence(output)

        return answer

    def managePreviousModel(self):
        """ Restore or reset the model, depending of the parameters
        If the destination directory already contains some file, it will handle the conflict as following:
         * If --reset is set, all present files will be removed (warning: no confirmation is asked) and the training
         restart from scratch (globStep & cie reinitialized)
         * Otherwise, it will depend of the directory content. If the directory contains:
           * No model files (only summary logs): works as a reset (restart from scratch)
           * Other model files, but modelName not found (surely keepAll option changed): raise error, the user should
           decide by himself what to doi
           * The right model file (eventually some other): no problem, simply resume the training
        In any case, the directory will exist as it has been created by the summary writer
        Args:
            sess: The current running session
        """

        print('WARNING: ', end='')

        model_path = os.path.join(os.path.curdir, 'save/model', self.args.corpus_name)

        ckpt = tf.train.latest_checkpoint(model_path)
        if ckpt:
            print('Restoring previous model from {}'.format(ckpt))
            self.saver.restore(self.sess, ckpt)  # Will crash when --reset is not activated and the model has not been saved yet


    def save_session(self, sess, step):

        tqdm.write('Checkpoint reached: saving model (don\'t stop the run)...')
        model_name = os.path.join('save/model', self.args.corpus_name, 'model.ckpt')
        self.saver.save(sess, model_name, global_step=step)  # TODO: Put a limit size (ex: 3GB for the modelDir)
        tqdm.write('Model saved.')


if __name__ == '__main__':
    chatbot = ChatBot()
