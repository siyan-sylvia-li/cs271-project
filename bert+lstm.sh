#!/bin/bash

python ft_pheno.py --exp_name bert-base+lstm --ft_mode classifier --lstm_head --batch_size 8 --epochs 50 --bert_name bert-base-cased --lr 1e-4 --seed 42 --early_stopping
python ft_pheno.py --exp_name bert-base+lstm --ft_mode lastbert --lstm_head --batch_size 8 --epochs 50 --bert_name bert-base-cased --lr 1e-4 --seed 42 --early_stopping
python ft_pheno.py --exp_name bert-base+lstm --ft_mode all --lstm_head --batch_size 8 --epochs 50 --bert_name bert-base-cased --lr 1e-4 --seed 42 --early_stopping
