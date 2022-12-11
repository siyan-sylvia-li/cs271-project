import torch
from transformers import BertConfig, BertTokenizer, BertModel

# https://huggingface.co/huggingfans/bert-mini/blob/main/config.json with modified vocab_size
BERT_CONFIG_FILE = 'config/bert-mini.json'
NUM_LSTM_LAYERS = 2


class PhenoPredictor(torch.nn.Module):
    def __init__(self, n_class, bert_name, use_pretrained=True, lstm_head=False, avg_cls=False, agg_sigmoid=False):
        super().__init__()
        self.n_class = n_class
        if use_pretrained:
            self.bert = BertModel.from_pretrained(bert_name)
            #self.bert.config.save_pretrained(save_directory='config')
        else:
            # Create BERT model without using pretrained weights.
            config = BertConfig.from_pretrained(BERT_CONFIG_FILE)
            self.bert = BertModel(config)
        if lstm_head:
            self.lstm_ensemble = torch.nn.LSTM(
                input_size=self.bert.config.hidden_size,
                hidden_size=self.bert.config.hidden_size,
                num_layers=NUM_LSTM_LAYERS)
        else:
            self.lstm_ensemble = None
        self.agg_sigmoid = agg_sigmoid  # if true, use the combination of sigmoids
        self.avg_cls = avg_cls    # if true, just average cls tokens
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.n_class)
        self.sigmoid = torch.nn.Sigmoid()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        # print(x['input_ids'].shape)
        final_output = []
        for i in range(x['input_ids'].shape[0]):
            # iterate through the batches
            o_len = x['orig_len'][i]
            # print(o_len)
            inp_ids = x['input_ids'][i, :o_len, :].to(self.device)
            att_masks = x['attention_mask'][i, :o_len, :].to(self.device)
            tok_type = x['token_type_ids'][i, :o_len, :].to(self.device)
            outputs = self.bert(input_ids=inp_ids, attention_mask=att_masks, token_type_ids=tok_type)
            # print(outputs[0].shape)
            cls_tokens = outputs[0][:, 0, :].squeeze(1)
            # cls_tokens = torch.mean(cls_tokens, dim=0)
            # print("CLS TOKEN 1", cls_tokens.shape)
            if self.lstm_ensemble is not None:
                output_cls_token, (hn, cn) = self.lstm_ensemble(cls_tokens)
                # print('LSTM output shape:', output_cls_token.shape)
                cls_tokens = output_cls_token[-1:]  # Take token at last time step.
            if self.avg_cls:
                cls_tokens = torch.mean(cls_tokens, dim=0)
            cls_tokens = self.classifier(cls_tokens)
            cls_tokens = self.sigmoid(cls_tokens)
            # print("CLS TOKEN 2", cls_tokens.shape)
            if self.agg_sigmoid:
                cls_tokens = cls_tokens - 0.5
                cls_tokens = self.sigmoid(cls_tokens)
            if not self.avg_cls:
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




