# Deep Q&A

#### Table of Contents

* [Presentation](#presentation)
* [Installation](#installation)
* [Running](#running)
* [Results](#results)
* [Pretrained model](#pretrained-model)

## Presentation

This work tries to reproduce the results of [A Neural Conversational Model](http://arxiv.org/abs/1506.05869) (aka the Google chatbot). It uses a RNN (seq2seq model) for sentence predictions. It is done using python and TensorFlow. It's belongs to the [csdn blog](http://blog.csdn.net/Irving_zhang/article/details/79088143)

For now, DeepQA support the following dialog corpus:
 * [Cornell Movie Dialogs](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) 


## Installation

The program requires the following dependencies (easy to install using pip: `pip3 install -r requirements.txt`):
 - pandas==0.22.0
 - tqdm==4.19.5
 - tensorflow==1.0.0
 - nltk==3.2.5
 - numpy==1.13.3

You might also need to download additional data to make nltk work.

```
python3 -m nltk.downloader punkt
```

## Running

### Chatbot

To train the model, simply run `training.py`. Once trained, you can test the results with `training.py with self.test = 'interactive' in args.py file.

## Results

It's possible to get some results after only 20 hours of training (on a CPU),
Of course, the network won't be really chatty:

Q: i like you.
A: You're not.

Q: i really like you.
A: Yeah?

Q: jack chen
A: Hi.

Q: what time is it?
A: Two months.

Q: what's the time?
A: I do n't know.

Q: you are cute.
A: I'm not.



## Pretrained model

You can find a pre-trained model 
[checkpoint.tar.gz](https://drive.google.com/open?id=1dSmFy52pW3j8CV1oSUqGQbjUq9FN_lKu).
[samples](https://drive.google.com/open?id=1AUAIVPu8MTIxfoWVnc9r6GO9b03OXE9n)
 To use it:
 1. Extract the zip file checkpoint.tar.gz inside `Seq2seq_from_scratch/save/model/cornell/
 2. Extract the zip file samples.tar.gz inside `Seq2seq_from_scratch/samples/cornell
 3. Run `./training.py with with self.test = 'interactive' in args.py file.




