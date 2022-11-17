import torch
from transformers import BertTokenizer, BertModel


class PhenoPredictor(torch.nn.Module):
    def __init__(self, n_class, bert_name):
        super().__init__()
        self.n_class = n_class
        self.bert = BertModel.from_pretrained(bert_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.n_class)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # print(x['input_ids'].shape)
        final_output = []
        for i in range(x['input_ids'].shape[0]):
            # iterate through the batches
            o_len = x['orig_len'][i]
            # print(o_len)
            inp_ids = x['input_ids'][i, :o_len, :]
            att_masks = x['attention_mask'][i, :o_len, :]
            tok_type = x['token_type_ids'][i, :o_len, :]
            outputs = self.bert(input_ids=inp_ids, attention_mask=att_masks, token_type_ids=tok_type)
            # print(outputs[0].shape)
            cls_tokens = outputs[0][:, 0, :].squeeze(1)
            # cls_tokens = torch.mean(cls_tokens, dim=0)
            # print("CLS TOKEN 1", cls_tokens.shape)
            cls_tokens = self.classifier(cls_tokens)
            cls_tokens = self.sigmoid(cls_tokens)
            # print("CLS TOKEN 2", cls_tokens.shape)
            cls_tokens = torch.mean(cls_tokens, dim=0)
            final_output.append(cls_tokens.unsqueeze(0))
            # print("CLS TOKEN", cls_tokens.shape)

        # all_hiddens.append(outputs[0][0].unsqueeze(0)) # Appending CLS token embeddings
        # for now we average the hidden states
        # fin_hiddens = torch.cat(all_hiddens, dim=0)
        # fin_hiddens = torch.mean(fin_hiddens, dim=0)
        # assert fin_hiddens.shape[-1] == self.bert.config.hidden_size
        final_output = torch.vstack(final_output)
        # print("FINAL OUTPUT", final_output.shape)
        return final_output




