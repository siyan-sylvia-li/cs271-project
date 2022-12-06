python3 ft_pheno.py --lr=0.00001 --ft_mode=classifier --batch_size=8 --bert_name=michiyasunaga/BioLinkBERT-base --exp_name=biolink_bert+lstm --lstm_head --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=lastbert --batch_size=8 --bert_name=michiyasunaga/BioLinkBERT-base --exp_name=biolink_bert+lstm --lstm_head --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=all --batch_size=8 --bert_name=michiyasunaga/BioLinkBERT-base --exp_name=biolink_bert+lstm --lstm_head --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=classifier --batch_size=8 --bert_name=michiyasunaga/BioLinkBERT-base --exp_name=biolink_bert --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=lastbert --batch_size=8 --bert_name=michiyasunaga/BioLinkBERT-base --exp_name=biolink_bert --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=all --batch_size=8 --bert_name=michiyasunaga/BioLinkBERT-base --exp_name=biolink_bert --early_stopping

