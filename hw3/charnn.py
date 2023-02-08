import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator
import torch.nn.functional as F


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======

    unique_chars = set(text)

    # Create the char to index mapping
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}

    # Create the index to char mapping
    idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}

    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======

    text_clean = [char for char in text if char not in chars_to_remove]

    # Use the join function to convert the list of characters back into a string
    text_clean = "".join(text_clean)

    # Count the number of characters that were removed
    n_removed = len(text) - len(text_clean)

    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======

    num_chars = len(char_to_idx)

    # Initialize a tensor of zeros
    result = torch.zeros((len(text), num_chars), dtype=torch.int8)

    # Set the appropriate element in the tensor to 1 for each character in the text
    for idx, char in enumerate(text):
        result[idx, char_to_idx[char]] = 1

    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======

    char_indices = ((embedded_text == 1).nonzero(as_tuple=False))[:, 1]

    # Use the index to look up the corresponding character in the idx_to_char mapping
    result = [idx_to_char[idx.item()] for idx in char_indices]

    # Use the join function to convert the list of characters back into a string
    result = "".join(result)

    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======

    num_samples = len(text[:-1]) // seq_len
    one_hot_text = chars_to_onehot(text, char_to_idx)
    num_relevant_chars = num_samples * seq_len
    samples = one_hot_text[:num_relevant_chars, :]
    samples = samples.reshape(num_samples, seq_len, -1).to(device)
    labels = one_hot_text[1:num_relevant_chars + 1, :]
    labels = ((labels == 1).nonzero(as_tuple=False))[:, 1]
    labels = labels.reshape(num_samples, seq_len).to(device)

    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    y /= temperature
    result = F.softmax(y, dim)

    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======

    # Disable gradient tracking
    with torch.no_grad():
        start_seq = chars_to_onehot(start_sequence, char_to_idx).unsqueeze(0).type(torch.float).to(device)
        # Forward pass
        y, hidden_state = model(start_seq)
        # Get last output char
        y_char = y.squeeze(0)[-1, :]
        # Sample from the output distribution
        y = torch.multinomial(hot_softmax(y_char, -1, T), 1).item()
        # Add char to output text
        out_text += idx_to_char[y]




        # Generate characters one by one
        while len(out_text)<n_chars:
            y_m = chars_to_onehot(out_text, char_to_idx).unsqueeze(0).type(torch.float).to(device)
            # Forward pass
            y, hidden_state = model(y_m,hidden_state)

            # Get last output char
            y_char= y[0, -1, :]

            # Sample from the output distribution
            y= torch.multinomial(hot_softmax(y_char, -1, T), 1).item()

            # Add char to output text
            out_text += idx_to_char[y]



    # ========================
    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======

        size = (len(self.dataset) // self.batch_size) * self.batch_size
        idx = torch.arange(size)
        idx = idx.reshape((self.batch_size, -1)).transpose(0, 1).flatten()

        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        """"
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        l_in_dim = self.in_dim

        for layer in range(n_layers):
            self.fc_hz = nn.Linear(in_features=self.h_dim, out_features=self.h_dim, bias=True)
            self.layer_params.append(self.fc_hz)
            self.fc_hr = nn.Linear(in_features=self.h_dim, out_features=self.h_dim, bias=True)
            self.layer_params.append(self.fc_hr)
            self.fc_hg = nn.Linear(in_features=self.h_dim, out_features=self.h_dim, bias=True)
            self.layer_params.append(self.fc_hg)
            self.fc_xz = nn.Linear(in_features=l_in_dim, out_features=self.h_dim, bias=False)
            self.layer_params.append(self.fc_xz)
            self.fc_xr = nn.Linear(in_features=l_in_dim, out_features=self.h_dim, bias=False)
            self.layer_params.append(self.fc_xr)
            self.fc_xg = nn.Linear(in_features=l_in_dim, out_features=self.h_dim, bias=False)
            self.layer_params.append(self.fc_xg)

        self.fc_hy = nn.Linear(in_features=self.h_dim, out_features=out_dim, bias=True)
        self.layer_params.append(self.fc_hy)

        """
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        layer_in_dim = self.in_dim
        for layer in range(n_layers):
            layer_dict = dict()
            layer_dict['hz'] = nn.Linear(in_features=self.h_dim, out_features=self.h_dim, bias=True)
            layer_dict['hr'] = nn.Linear(in_features=self.h_dim, out_features=self.h_dim, bias=True)
            layer_dict['hg'] = nn.Linear(in_features=self.h_dim, out_features=self.h_dim, bias=True)
            layer_dict['xz'] = nn.Linear(in_features=layer_in_dim, out_features=self.h_dim, bias=False)
            layer_dict['xr'] = nn.Linear(in_features=layer_in_dim, out_features=self.h_dim, bias=False)
            layer_dict['xg'] = nn.Linear(in_features=layer_in_dim, out_features=self.h_dim, bias=False)

            self.layer_params.append(layer_dict)
            for key, val in layer_dict.items():
                self.add_module(name=f"layer{layer}_" + key, module=val)
            layer_in_dim = self.h_dim

        hy = nn.Linear(in_features=layer_in_dim, out_features=self.out_dim, bias=True)
        self.layer_params.append(hy)
        self.add_module(name="hy", module=hy)

        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        """""
        layer_output = []
        i = 0
        for t in range(seq_len):
            prev_hidden_state = torch.stack(layer_states, dim=1)
            layer_states = []
            x_curr = layer_input[:, t, :]

            for k in (self.layer_params):
                h_prev = prev_hidden_state[:, i, :]

                z = self.sig((self.fc_xz(x_curr) + self.fc_hz(h_prev)))

                r = self.sig(self.fc_xr(x_curr) + self.fc_hr(h_prev))
                g = self.tanh(self.fc_xg(x_curr) + self.fc_hg(r * h_prev))
                h = z * h_prev + (1 - z) * g
                layer_states.append(h.clone())
                x_curr = self.dropout(h)
                i += 1

            layer_output.append(x_curr)
        hidden_state = torch.stack(layer_states, dim=1)
        layer_output = torch.stack(layer_output, dim=1)
        """""

        layer_output = []
        for t in range(seq_len):
            prev_hidden_state = torch.stack(layer_states, dim=1)
            layer_states = []
            curr_x = input[:, t, :]

            for k, layer in enumerate(self.layer_params):
                if isinstance(layer, dict):
                    h_k = prev_hidden_state[:, k, :]
                    z = self.sigmoid(layer['xz'](curr_x) + layer['hz'](h_k))
                    r = self.sigmoid(layer['xr'](curr_x) + layer['hr'](h_k))
                    g = self.tanh(layer['xg'](curr_x) + layer['hg'](r * h_k))
                    h = z * h_k + (1 - z) * g
                    layer_states.append(h.clone())
                    curr_x = self.dropout(h)
                else:
                    layer_output.append(layer(curr_x))
        hidden_state = torch.stack(layer_states, dim=1)
        layer_output = torch.stack(layer_output, dim=1)

        # ========================
        return layer_output, hidden_state
