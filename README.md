This is theDeep Transition Dependency Parser I made for my Natural Language Processing class using PyTorch. 
It is based around an LSTM (long short-term memory) neural network. 
All of the code and training data are included in this repo.
Note that it is  easier to follow the explanations and see the results of the code by reading pset3.ipynb.
I recommend looking at pset3.ipynb before anything else, as all other code is used to generate the models that can be seen in the Jupyter notebook.

Results of the graded tests are featured below.
![alt text](https://raw.githubusercontent.com/smikhaylov3/Deep-Transition-Dependency-Parser/master/Dependency%20parsing%20results.jpg)

A summary of the general control flow of the program follows below.

->Initialize your parsing stack and input buffer.
->At each step, until the parse is done:
  ->Extract some features. We will start with simple features, but these can be anything: words in the sentence, the configuration of the stack, the configuration of the input buffer, the previous action, etc.
  ->Send these features through a feed-forward (FF) network to get a probability distribution over actions (SHIFT, ARC_L, ARC_R). The next action you choose is the one with the highest probability.
  ->If the action is an arc- operation, you use a neural network to combine the two items in the operation and get a dense output to place back on the input buffer.
  
  The most important classes in the program are:
  
->Feature extraction in feat_extractors.py
->The ParserState class, which keeps track of the input buffer and parse stack, and offers a public interface for doing the parsing actions to update the state
->The TransitionParser class, which is a PyTorch module where the core parsing logic resides, in parsing.py.
->The neural network components in neural_net.py

The components of the neural network are separated like this:

->TransitionParser, the base component that contains and coordinates the other substitutable components
->Embedding Lookup: You will implement three flavors of embeddings. These embeddings are used to initialize the input buffer, and will be shifted on the stack / serve as inputs to the combiner networks.
->VanillaWordEmbedding just gets embeddings from a lookup table.
->BiLSTMWordEmbedding will run a sequence model in both directions over the sentence. The hidden state at step t is the embedding for the t-th word of the sentence.
->SuffixAndWordEmbedding gets embeddings for words as in the vanilla embeddings, and also gets embeddings for word suffixes, and concatenates t
->Action Choosing: You will implement two action choosing components:
->FFActionChooser is a simple feed-forward neural network that outputs log probabilities over the three actions given the extracted features as input.
->LSTMActionChooser applies a sequence model that takes the hidden state of the previous action decision as input.
->Combiners: You will implement two combiners, which are the network components that take the two embeddings of the items in an arc- operation and creates a single vector.
->FFCombiner takes the two input embeddings and gives a dense output
->LSTMCombiner applies a sequence model, where the output embedding is the hidden state of the next timestep.
