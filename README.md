# **Neural Machine Translation**

This tutorial demonstrates how to train a sequence-to-sequence (seq2seq) model for Spanish-to-English translation using [Bahdanau’s Attention](https://arxiv.org/abs/1409.0473) and [Luong’s Attention](https://arxiv.org/abs/1508.04025)

The sequence-to-sequence (seq2seq) model with attention is a type of neural network architecture that is commonly used for tasks such as machine translation and speech recognition. The model consists of an encoder and a decoder, which are connected by an attention mechanism.

The encoder takes in a sequence of input tokens and produces a fixed-length vector representation of the input sequence, which contains all the information necessary to generate the output sequence. The encoder is typically implemented as a recurrent neural network (RNN), such as a long short-term memory (LSTM) or a gated recurrent unit (GRU).

The decoder takes the encoder output vector and generates the output sequence one token at a time. At each time step, the decoder RNN takes in the previous output token, the current state of the decoder RNN, and the encoder output vector. The decoder RNN generates a probability distribution over the possible output tokens, and samples the next token from this distribution.

The attention mechanism allows the decoder to selectively focus on different parts of the input sequence at each time step. Specifically, the attention mechanism computes a weighted sum of the encoder output vectors, where the weights are learned based on the current state of the decoder RNN. These weights indicate how much attention the decoder should pay to each part of the input sequence when generating the output at the current time step.

During training, the model is trained to minimize the cross-entropy loss between the predicted output sequence and the true output sequence. During inference, the model generates the output sequence by iteratively sampling the most likely output token at each time step, conditioned on the previous output tokens and the encoder input sequence.

![](https://github.com/MarwanMohamed95/Machine-Translation-with-Attention/blob/main/attention_mechanism.jpg?raw=true)


Bahdanau’s Attention:
------------------------
$$
\begin{aligned}
e_{t, t'} &= v_a^T \cdot \tanh(W_a \cdot h_t + U_a \cdot \tilde{h}_{t'}) \\
\alpha_{t, t'} &= \frac{\exp(e_{t, t'})}{\sum_{k=1}^T \exp(e_{t, k})} \\
c_t &= \sum_{t'=1}^T \alpha_{t, t'} \cdot \tilde{h}_{t'}
\end{aligned}
$$

Luong's Attention:
---------------------
$\large a_{i,j} = \frac{\exp(score(h_i, \bar{h}_j))}{\sum_{k=1}^{T_y} \exp(score(h_i, \bar{h}_k))}$

where
$\large score(h_i, \bar{h}_j) = h_i^\top W_a \bar{h}_j$

and
$\large \bar{h}_j = \frac{1}{T_x} \sum_{i=1}^{T_x} h_i$

where $h_i$ is the encoder hidden state at time step $i$,
$\bar{h}_j$ is the mean of all encoder hidden states,
$T_x$ is the number of encoder time steps, and
$T_y$ is the number of decoder time steps.
