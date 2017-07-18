# Keras External Embeddings

## Problem
Keras supports using [pretrained word embeddings](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) in your models. In a lot of cases, it makes sense to **freeze** the pretrained word embeddings at training time. Keras provides an easy option for that in its [Embedding](https://keras.io/layers/embeddings/) layer, by setting the `trainable` argument to False (check the [FAQ section](https://keras.io/getting-started/faq/#how-can-i-freeze-keras-layers)).

However, by adding the Embedding layer to your model, you will be saving the word embeddings alongside your model. This is fine if you are dealing with a couple of models. In production environments, however, you might have several models, all using frozen pretrained embeddings. In that case, you will be duplicating the embeddings in all models. This results in orders of magnitude increase in storage on disk and in much higher RAM usage. It is more efficient to share the embeddings across the models and to perform the mapping from words to vectors only once for all your models.

This repository shows how this can be done by building on the [same example](https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py) provided with Keras with GloVe embeddings and the 20 Newsgroup dataset. The first file `pretrained_word_embeddings.py` is the original file from Keras. The second file `pretrained_external_word_embeddings.py` is the one where the embeddings are external to the model. The main changes are in how the data is loaded and in the first layer of the model. 

To run it yourself, head to the files and adjust the directories `GLOVE_DIR` and `TEXT_DATA_DIR` to your preference, along with other parameters.
Then simply run `python pretrained_external_word_embeddings.py`

## Prerequisites:
* [Keras](https://github.com/fchollet/keras)

     
## Developer
[Hamza Harkous](http://hamzaharkous.com)

## License
[MIT](https://opensource.org/licenses/MIT)