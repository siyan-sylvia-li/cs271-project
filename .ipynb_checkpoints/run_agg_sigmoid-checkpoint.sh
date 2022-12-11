#!/bin/bash

python3 ft_pheno.py --lr=0.00001 --ft_mode=classifier --batch_size=8 --bert_name=bert-base-cased --exp_name=bert-base+agg_sigmoid --agg_sigmoid --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=lastbert --batch_size=8 --bert_name=bert-base-cased --exp_name=bert-base+agg_sigmoid --agg_sigmoid --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=all --batch_size=8 --bert_name=bert-base-cased --exp_name=bert-base+agg_sigmoid --agg_sigmoid --early_stopping

python3 ft_pheno.py --lr=0.00001 --ft_mode=classifier --batch_size=8 --bert_name=michiyasunaga/BioLinkBERT-base --exp_name=biolink_bert+agg_sigmoid --agg_sigmoid --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=lastbert --batch_size=8 --bert_name=michiyasunaga/BioLinkBERT-base --exp_name=biolink_bert+agg_sigmoid --agg_sigmoid --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=all --batch_size=8 --bert_name=michiyasunaga/BioLinkBERT-base --exp_name=biolink_bert+agg_sigmoid --agg_sigmoid --early_stopping

python3 ft_pheno.py --lr=0.00001 --ft_mode=classifier --batch_size=8 --bert_name=emilyalsentzer/Bio_ClinicalBERT --exp_name=clinical-biobert+agg_sigmoid --agg_sigmoid --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=lastbert --batch_size=8 --bert_name=emilyalsentzer/Bio_ClinicalBERT --exp_name=clinical-biobert+agg_sigmoid --agg_sigmoid --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=all --batch_size=8 --bert_name=emilyalsentzer/Bio_ClinicalBERT --exp_name=clinical-biobert+agg_sigmoid --agg_sigmoid --early_stopping

python3 ft_pheno.py --lr=0.00001 --ft_mode=classifier --batch_size=8 --bert_name=emilyalsentzer/Bio_Discharge_Summary_BERT --exp_name=discharge_bert+agg_sigmoid --agg_sigmoid --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=lastbert --batch_size=8 --bert_name=emilyalsentzer/Bio_Discharge_Summary_BERT --exp_name=discharge_bert+agg_sigmoid --agg_sigmoid --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=all --batch_size=8 --bert_name=emilyalsentzer/Bio_Discharge_Summary_BERT --exp_name=discharge_bert+agg_sigmoid --agg_sigmoid --early_stopping

python3 ft_pheno.py --lr=0.00001 --ft_mode=classifier --batch_size=8 --bert_name=allenai/scibert_scivocab_cased --exp_name=scibert+agg_sigmoid --agg_sigmoid --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=lastbert --batch_size=8 --bert_name=allenai/scibert_scivocab_cased --exp_name=scibert+agg_sigmoid --agg_sigmoid --early_stopping
python3 ft_pheno.py --lr=0.00001 --ft_mode=all --batch_size=8 --bert_name=allenai/scibert_scivocab_cased --exp_name=scibert+agg_sigmoid --agg_sigmoid --early_stopping
