# Fine tuning of a BERT model aimed at the sentiment analysis of tweets in the Italian language

## BERT

BERT (Bidirectional Encoder Representations from Transformers), released at the end of 2018, is
a method for pre-training linguistic representations. In fact, it is
possible to use these models to extract linguistic features of
high quality from textual data, or you can refine these models on
a specific task (classification, entity recognition, question
answering, etc.). The use of transformers in this type of task is very
convenient in that they manage, during the learning phases
assign different weights to the various components of the input (in the case of
BERT the input is represented by natural language sentences).

One of the most important components of the Bert models is the _Tokenizer_. This is
a sub-component, the purpose of which is to fragment the input sentences into
input sentences in the form of tokens, i.e. atomic components of the language that
allow Bert to easily identify words with similar meanings.
assimilated meanings. Below is an example of tokenization
carried out by Bert.

As the purpose of our project was to perform a
analysis task on a set of tweets in the Italian language, it was
necessary to rely on a tokenizer and a model pre-trained on this
language. After trying several alternatives, including the " **_bert-base-italian-
xxl-cased_** " and " **_gilberto-uncased-from-camembert_** " the template was chosen
"model was chosen, as the latter proved to be the best in terms of accuracy.
proved to be the best in terms of accuracy and training speed
(number of trainable parameters).
Below are links to the HuggingFace and GitHub pages where it is
can be found each pre-trained model and the corresponding tokenizer
used.

- https://huggingface.co/dbmdz/bert-base-italian-xxl-cased
- https://huggingface.co/m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0
- https://huggingface.co/idb-ita/gilberto-uncased-from-camembert

## Goal and Dataset used

As mentioned above, the objective of this project is to do a
sentiment analysis on a particular topic, namely the Russian-Ukrainian conflict, thanks to the analysis of tweets in the Italian language.
Ukrainian conflict, thanks to the analysis of tweets in the Italian language.

Considering the availability of full-bodied datasets in this regard, we want to extract
the sentiment from a sample, since usually on social
network, users give their opinions without filters and restrictions (hence,
more truthful opinions). The dataset was constructed from several files
available on Kaggle (link 1 and link 2), characterised by tweets in different languages
different languages, content with non-textual components (emoji, hashtags), and
useless information (such as user id, name, time, etc.). Therefore, a
data processing had to be carried out, as the individual
individual files in 'csv' format, they were scanned to detect the names of the
columns, and records were selected that had 'it' (code
identifier) as the language (column 'language'). As the last step of the ETL
only the column corresponding to the tweet was selected ('text' column),
resulting in a cleaned dataset (Figure 1).

However, as it was not possible to find a dataset labelled on
this specific topic, it was decided to train the model on a different dataset
different dataset, within which there are also tweets in the Italian language, dating
different time periods (2011 - 2017). For each record, a
label: 0, if it is perceived as a negative tweet, 1 if neutral, 2 in the case of
positive sentiment. Following this, we proceeded with the
ETL phase.

## Preprocessing of labelled data and splitting

Once the datasets necessary for the training and testing phases of our model had been found, the
our model, the next step was to cleanse the data of all impurities and superfluous information.
impurities and superfluous information that could have negatively influenced the final result.
negatively affect the final result.

These data cleaning operations were carried out with the support of the
library " _Pandas_ "; the latter allows for very agile execution of
manipulation operations on datasets represented through the
DataFrame format.

The first step was to import the raw database, exploiting the
datasets' library offered by " _HuggingFace_ ". The latter allowed us
allowed us to import the data directly into the Colab environment, without having to
time-consuming downloading and uploading to Drive.

Once the full database was loaded, the actual manipulation operations began.
manipulation operations began. The starting dataset was divided into three partitions
into three partitions: training, validation and test. Aware of the fact that we would
that we would have to perform operations on the entire dataset, the first operation we
operation was to concatenate the sub-components of the dataset converted
into Pandas DataFrame.

Having converted the data into a much more manageable format, the second step was to
was to apply the actual cleaning functions. Specifically,
the procedure we applied is contained entirely in the function
clean_text() function; the latter, as a first operation, applies a filter to the sentences
sentences given as input by eliminating all special characters (those that do not
belong to the set a-z, A-Z, ".", "?", "!", ",") and punctuation marks;
having done so, all components that do not have to do with the true message are also removed from the sentence.
to do with the actual message, e.g. the header
http header and the string user (present in all tweets, it indicates the authors of the post).

Before finishing, the function also takes care of removing all special symbols
special symbols such as emoji, map symbols and IoS flags. Having done so, we find ourselves with
a dataset cleaned of all impurities and containing only the
information related to the actual messages.

Once these cleaning operations were completed, the only thing left to do was to
was to re-partition the dataset into the three partitions training
validation and test partitions. Due to the limited amount of data available,
we chose to allocate most of it to the training phase,
by performing a 90/5 split. In fact, the tweets can be of various kinds and
very different topics, consequently the model with limited data
data may fail to capture the possible nuances (even ironic) that may characterise the tweets.
that may characterise the tweets.

## Tokenizer and training phase

Having performed the split, the next step was to prepare the dataframes
which would later be given as input to the Bert model; to do this, the tweets
were processed by the tokenizer which transformed them into groups of tokens
(interpretable by the Bert model).

Below is an example of tokenization by the model
chosen (Figure 7); as can be observed
despite the cleaning processes, it is not possible to eliminate 100% of all the
meaningless syntagmas; this is due to the very nature of the
dataset itself, which, containing real posts, written by real people, inevitably contain
inevitably contain lexical inaccuracies and mistakes. In addition, the
special' tokens are added, such as [CLS] which indicates the beginning of the sentence [SEP] which
indicates the end of the tweet and [PAD] which specifies tokens for padding (in order to
reach the maximum length set in the initial phase).

Once the dataset was ready, we proceeded with the definition of the parameters
fundamental parameters for fine-tuning the pre-padded Bert model.

By experimenting with different values, we determined that the
(in terms of accuracy) were best when setting:
1. **Learning rate** : $3\times10^{-5}$
2. **Training epochs** : 5
3. **Batch size** : 32
4. **Max Length** : 11

Actually, the number of epochs represents a higher limitation, as
as it was clear from the outset that the model reached its
maximum accuracy after two or three epochs at the most. For this reason, it was
provision was made to save only the best model.

As far as the DataLoader is concerned, it must know the size of the batch
for training, after which the DataLoader for training and
the one for validation, as the model requires tensors as input.
Before proceeding with the actual fine-tuning, it must be considered that
BERT requires specifically formatted input. For each input sentence
tokenized previously, it is necessary to create:

- "input ids": a sequence of integers that identifies each token of
    input with its index number in the tokenizer's vocabulary;
- "attention mask": a sequence of 1 and 0, with 1 for all input tokens and 0 for all
    input tokens and 0 for all padding tokens;
- "labels": a label associated with the tokenized phrase.

To modify the pre-padded model, a series of interfaces are used,
provided by Huggingface PyTorch and in this case a model is used
BERT model with a single linear layer added for classification, called
BertForSequenceClassification. As the input data is provided,
the entire pre-trained BERT model and the additional classification layer
are trained for our specific task. Since the pre-trained layers
trained layers already encode a lot of information about the language, training
of the classifier is relatively inexpensive. Rather than training from
from scratch all the layers of a large model, it is as if we were to
train only the top layer, with a few modifications to the lower layers
to adapt them to our task, especially since it is an SA task. At this
point we can proceed with the training, which consists of several steps;
in detail:

- Training cycle:
    -- You instruct the model to calculate gradients by setting the model in training mode.
    -- Data inputs and labels are unpacked
    -- Upload data to the GPU for acceleration.
    -- The gradients calculated in the previous step are deleted. In pytorch, gradients accumulate by default,
    unless you explicitly delete them.
    -- Forward pass, i.e. you give the input data to the network
    -- Backward pass (backpropagation)
    -- Telling the network to update parameters (optimiser.step)
    -- Tracking variables to monitor progress

- Evaluation cycle:
    -- You tell the model not to calculate gradients by setting the model in evaluation mode.
    -- Data inputs and labels are unpacked
    -- Data is uploaded to the GPU for acceleration
    -- Forward pass, i.e. input data is given to the network
    -- Loss calculation on validation data and tracking of variables to monitor progress.

In order to evaluate the performance of the models produced, the
library " _Evaluate"_ , offered by HuggingFace. This has within it
dozens of evaluation metrics aimed at estimating the goodness of various
machine learning models. In our case, the only metric used was
In our case, the only metric used was the _accuracy_, which consists of the ratio between the correct predictions and the
total number of samples. In addition to monitoring the loss of
In addition to monitoring the training loss, the validation loss was also considered, in order to understand
if underfitting occurred, i.e. a model that fails to generalise
(e.g. due to noise or an unsuitable dataset), or overfitting (the model specialises too much on data in
overfitting (the model specialises too much on input data, so that in validation
insufficient results are obtained). As can be seen from Figure 8 Figure 9 ,
the training loss decreases over time, while the validation loss
increases, showing that the model is overfitting.

The maximum accuracy is only reached at the third
epoch and is 73.78% (Figure 9).

Following the training phase, it can be stated that the results obtained
are not as optimal as in other tasks. However, this is due to the fact that the
pre-trained model gets into difficulties, as similar tweets may have
different labels (often, a negative news item may also be reported
by a newspaper, with a more neutral tinge). In addition, tweets
may be written in incorrect Italian.

## Testing phase and results

Once the final model is obtained, testing is performed on the partition
dedicated partition, performing the same steps as before. Therefore,
using the same tokenizer, we feed the prepared data to the
model. The number of tweets processed is 152 and an accuracy of 73% is obtained,
indicating results in line with the previous phase.

From the confusion matrix (Figure 10), it is confirmed that very often
the model fails to correctly predict the class in the negative and
neutral cases. However, one must realise that one has a small number
small number of test samples, so the percentages may be
different in the case of a larger number of data.

Once the labelled dataset was tested, we used the model on the
unlabelled dataset, created at an early stage. As previously
analysed, this dataset contained approximately 15,000 tweets related to the
war in Ukraine. Since no classes are available for each tweet,
it is not possible to calculate a useful metric; however, looking at the data, one can
note that in many cases the model predicts correctly (Figure 11). In the
In cases where there is ambiguity, it can be seen that the model does not effectively
actually discriminate between the different cases; for instance, the first
tweet, despite using the word 'pray', expresses negativity, yet the model
model interprets it as a neutral text. In contrast, in the third to last tweet
figure uses words such as "flowers" or "songs", in an
ironic key, consequently label 2 is associated, even though the tweet has an
overtly negative connotation.

At other times, it can be seen that the tweets originally featured many hashtags
and/or mentions of other users, so that the actual content may be
understood with greater difficulty. In addition, especially when having
long sentences, the sentiment is only clarified in the final part of the tweet (Figure
12), whereby, also due to the initially set maximum length
(11 characters), the model finds it difficult to assign the right class,
often opting for neutrality.

To conclude, having ascertained the presence of inaccurate predictions, the
counted the elements of each class and the following results were obtained:

As expected, the general sentiment is mostly negative or (7310 tweets)
neutral (7137 tweets), while the positive tweets are significantly less
(1102 in total). This is mainly due to the fact that on Twitter are
published (perceived as neutral), whereas it is the users
who usually write content with more negative notes.

## Conclusions and future developments

This project has been a way for us to deepen several
aspects of particular relevance in the field of Natural Language Processing,
such as knowledge of the BERT network and Sentiment Analysis. To see how
this type of techniques and technologies are applied in areas that
current topics, such as the war in Ukraine, made us really feel the
truly perceive the importance of the issues studied.

Despite the less than ideal behaviour of the trained models, we are
satisfied with the results achieved; in fact, if we take into consideration the fact
that the training dataset was not designed for what was later
the final application of the model produced, it could be said that the results
obtained are more than satisfactory.

In the future, wishing to further improve the performance
performance of the algorithm, one might think of setting up a dataset containing more specific and
more specific and more related to the test data (tweets related to the
Russian-Ukrainian conflict) and to perform a more punctual pre-processing, so as to
so as to exclude non-useful parts of the content.

#### _Disclaimer: in-text images refer to 'Relazione_BERT.pdf' file_
