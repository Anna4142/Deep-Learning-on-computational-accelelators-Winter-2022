r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['seq_len'] = 80
    hypers['h_dim'] = 128
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 1e-3
    hypers['lr_sched_factor'] = 1e-1
    hypers['lr_sched_patience'] = 4
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
Splitting the corpus into sequences is done because the network is designed to process sequential data,

such as time series data or natural language. Training on the whole text would not take advantage of this capability,

as the model would not be able to learn the dependencies between different parts of the text. Additionally, training

on the whole text would require a much larger amount of memory and computational resources, making it infeasible for most systems.

It can also lead to overfitting


"""

part1_q2 = r"""
**Your answer:**

It is possible for LSTMs to generate text that shows memory longer than the sequence length because of the hidden state

that allows them to maintain information over a long period of time.This allows the LSTM to maintain information from previous time steps,

even if the input sequence is longer than the specified sequence length. Additionally, we use "gates" to control the flow of information 

into and out of the memory cell, which allows the network to selectively retain or discard information as needed.
"""

part1_q3 = r"""
**Your answer:**

the network rely on the order of the input sequence to learn temporal dependencies. 

Shuffling the order of the batches would disrupt the temporal structure of the data and make it harder for the model to learn these dependencies.



Shuffling the order of the batches would cause the hidden states  to be reset at the beginning of each new batch,
 
which could make it difficult for the model to maintain information across batches. This could lead to poor performance,

especially when working with smaller batch sizes.
"""

part1_q4 = r"""
**Your answer:**


1)In the context of text generation , temperature is a hyperparameter that controls the randomness or "creativity" of the generated text.

The default temperature is typically set to 1.0. the temperature parameter controls the diffusion of the probability distribution 

of the tokens generated  during the sampling process. As the temperature gets higher, the distribution becomes more uniform

Lowering the temperature for sampling can make the generated text more predictable and less creative.

On the other hand, as the temperature gets lower, the distribution becomes more spiky, meaning that the model is more
 
 likely to generate tokens with higher probabilities. This can lead to more predictable and less creative text, but also to more coherent text.
 

2)When the temperature is very low, the model will tend to generate text that is similar to what it has seen during training. 

This is because the model will be mostly choosing the most likely token (the token with the highest probability) at each time step.



3)On the other hand, when the temperature is very high, the model will generate more creative and unpredictable text.

This is because the model will be more likely to choose tokens with lower probabilities, which will introduce more randomness
 
 and variability into the generated text. However, when the temperature is too high, the model may generate text that is nonsensical 
  
or doesn't make sense, because it is choosing highly unlikely tokens.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=4, h_dim=128, z_dim=256, x_sigma2=0.1, learn_rate=0.0005, betas=(0.9, 0.999), )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The hyperparameter x_sigma2 (also known as the reconstruction loss scaling factor) is used to adjust the weight of the 

reconstruction loss term in the VAE loss function. The VAE loss function is a combination of two parts: the reconstruction 

loss and the KL divergence loss.

x_sigma2 is used to scale the reconstruction loss term, and is typically set to the variance of the data. 

It controls the trade-off between the reconstruction loss and the KL divergence loss. A high value of x_sigma2 will result
 
 in a higher weight for the reconstruction loss term, which means the model will focus more on reconstructing the input data accurately.

A low value of x_sigma2 will result in a lower weight for the reconstruction loss term, which means the model will focus 

more on matching the latent space to the prior distribution.

"""

part2_q2 = r"""
**Your answer:**
1)The reconstruction loss give by 
 $$
L_{reconstruciton} =
\frac{1}{\sigma^2 d_x} \left\| \bb{x}- \Psi _{\bb{\beta}}\left(  \bb{\mu} _{\bb{\alpha}}(\bb{x})  +
\bb{\Sigma}^{\frac{1}{2}} _{\bb{\alpha}}(\bb{x}) \bb{u}   \right) \right\| _2^2
$$
measures the difference between the original input and the output generated by the decoder.
 
 The aim is to minimize this loss, so that the decoder can generate outputs that are as close as possible to the original inputs.

The KL divergence loss give by 
 $L_{KL}=\mathrm{tr}\,\bb{\Sigma} _{\bb{\alpha}}(\bb{x}) +  \|\bb{\mu} _{\bb{\alpha}}(\bb{x})\|^2 _2 - d_z - \log\det \bb{\Sigma} _{\bb{\alpha}}(\bb{x}),
$
measures the difference between the latent space distribution learned by the encoder and a prior distribution
 
(e.g. a normal distribution). The KL divergence loss term encourages the encoder to learn a latent space distribution
  
that is similar to the prior distribution.
  
2)The KL divergence loss term measures the difference between the latent space distribution learned by the encoder and a prior distribution 

(e.g. a normal distribution). The KL divergence loss term encourages the encoder to learn a latent space distribution that is similar to the prior distribution.
 
This means that if the prior distribution is a normal distribution, the encoder will be trained to produce a latent space distribution that is also
  
close to a normal distribution. 


3)The benefit of this effect is that it makes the VAE more robust to overfitting. By encouraging the latent space to have a certain desired
 
 structure, the model is less likely to memorize the training data and generalize better to unseen data.

Another benefit is that it allows for better exploration of the latent space, by constraining it to a more reasonable range of values.
 
this will ensure that the values in the latent space are not too extreme and can be easily mapped back to the original data space.

Additionally, it also allows for sampling new data from the latent space. By sampling from the prior distribution, the decoder
 
can generate new data that is similar to the training data, which is useful for tasks such as data generation and image synthesis.
"""

part2_q3 = r"""
**Your answer:**
In the Variational Autoencoder (VAE), we aim to maximize the evidence lower bound (ELBO) which is an approximation of the log-likelihood of the data. The ELBO is defined as:

ELBO = E[log p(x|z)] - D_KL(q(z|x) || p(z))


We start by maximizing the ELBO because it is a lower bound on the log-likelihood of the data. By maximizing this lower bound, we are also maximizing the log-likelihood
 
of the data. The KL divergence term acts as a regularizer, encouraging the approximate posterior to be similar to the prior.

The goal of the VAE is to learn a generative model that can generate data that is similar to the training data. 

By maximizing the ELBO, the VAE learns to balance the trade-off between accurately reconstructing the input data and matching the latent space to the prior distribution. 

This allows the VAE to generate high-quality data while avoiding overfitting.
"""

part2_q4 = r"""
**Your answer:**
modeling the log of the variance allows for more numerical stability and better interpretability, while also ensuring that
 
 the variance is always positive.
"""

# ==============

# ==============
# Part 3 answers




part3_q1 = r"""
**Your answer:**


"""

part3_q2 = r"""
**Your answer:**

"""

part3_q3 = r"""
**Your answer:**

"""

part3_q4 = r"""
**Your answer:**

"""


# ==============
