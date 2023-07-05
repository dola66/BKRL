import pdb

from model.BasicModule import *
class DynamicGRU(BasicModule):
    def __init__(self, input_dim, output_dim,
                 num_layers=1, bidirectional=False,
                 batch_first=True):
        super().__init__()
        self.embed_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.gru = nn.GRU(self.embed_dim,
                            self.hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=self.batch_first)


    def forward(self, inputs, lengths,one = False):

        if one == True:
            hidden = lengths
            out, ht = self.gru(inputs,hidden)

        else:

            _, idx_sort = torch.sort(lengths, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            sort_embed_input = inputs.index_select(0, Variable(idx_sort))

            sort_lengths = list(lengths[idx_sort])

            inputs_packed = nn.utils.rnn.pack_padded_sequence(sort_embed_input,
                                                              sort_lengths,
                                                              batch_first=True)

            out_pack, ht = self.gru(inputs_packed)




            out = nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
            out = out[0]


            ht = torch.transpose(ht, 0, 1)[idx_unsort]
            ht = torch.transpose(ht, 0, 1)

            out = out[idx_unsort]


        return out, ht




