python ft_pheno.py --exp_name bert-base --ft_mode classifier --batch_size 8 --epochs 50 --bert_name bert-base-cased --lr 1e-4 --seed 42 --early_stopping
python ft_pheno.py --exp_name bert-base --ft_mode lastbert --batch_size 8 --epochs 50 --bert_name bert-base-cased --lr 1e-4 --seed 42 --early_stopping
python ft_pheno.py --exp_name bert-base --ft_mode all --batch_size 8 --epochs 50 --bert_name bert-base-cased --lr 1e-4 --seed 42 --early_stopping
sudo shutdown