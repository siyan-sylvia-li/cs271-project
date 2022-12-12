python3 ft_pheno.py --lr=0.00001 --ft_mode=classifier --batch_size=8 --bert_name=allenai/scibert_scivocab_cased --exp_name=scibert+lstm --lstm_head --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=lastbert --batch_size=4 --bert_name=allenai/scibert_scivocab_cased --exp_name=scibert+lstm --lstm_head --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=all --batch_size=2 --bert_name=allenai/scibert_scivocab_cased --exp_name=scibert+lstm --lstm_head --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=classifier --batch_size=8 --bert_name=allenai/scibert_scivocab_cased --exp_name=scibert --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=lastbert --batch_size=4 --bert_name=allenai/scibert_scivocab_cased --exp_name=scibert --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=all --batch_size=2 --bert_name=allenai/scibert_scivocab_cased --exp_name=scibert --early_stopping