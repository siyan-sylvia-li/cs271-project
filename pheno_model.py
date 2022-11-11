import torch
from transformers import BertTokenizer, BertModel


class PhenoPredictor(torch.nn.Module):
    def __init__(self, n_class, bert_name):
        super().__init__()
        self.n_class = n_class
        self.bert = BertModel.from_pretrained(bert_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.n_class)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, xs):
        all_hiddens = []
        for x in xs:
            outputs = self.bert(**x)
            all_hiddens.append(outputs[0][0].unsqueeze(0)) # Appending CLS token embeddings
        # for now we average the hidden states
        fin_hiddens = torch.cat(all_hiddens, dim=0)
        fin_hiddens = torch.mean(fin_hiddens, dim=0)
        assert fin_hiddens.shape[-1] == self.bert.config.hidden_size
        out = self.classifier(fin_hiddens)
        out = self.softmax(out)
        return out




