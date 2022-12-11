python3 ft_pheno.py --lr=0.00001 --ft_mode=classifier --batch_size=2 --bert_name=emilyalsentzer/Bio_Discharge_Summary_BERT --exp_name=discharge_bert+lstm --lstm_head --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=lastbert --batch_size=2 --bert_name=emilyalsentzer/Bio_Discharge_Summary_BERT --exp_name=discharge_bert+lstm --lstm_head --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=all --batch_size=2 --bert_name=emilyalsentzer/Bio_Discharge_Summary_BERT --exp_name=discharge_bert+lstm --lstm_head --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=classifier --batch_size=2 --bert_name=emilyalsentzer/Bio_Discharge_Summary_BERT --exp_name=discharge_bert --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=lastbert --batch_size=2 --bert_name=emilyalsentzer/Bio_Discharge_Summary_BERT --exp_name=discharge_bert --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=all --batch_size=2 --bert_name=emilyalsentzer/Bio_Discharge_Summary_BERT --exp_name=discharge_bert --early_stopping

