export CUDA_VISIBLE_DEVICES=1

model=Informer

# ETTh1
python -u run.py \
  --model $model \
  --features M \
  --root_path ./data/ETT-small/ \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 0.0001 \
  --freq h \
  --d_model 512 \
  --d_ff 2048

# ETTh2
python -u run.py \
  --model $model \
  --features M \
  --root_path ./data/ETT-small/ \
  --data_path ETTh2.csv \
  --data ETTh2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 0.0001 \
  --freq h \
  --d_model 512 \
  --d_ff 2048

# ETTm1
python -u run.py \
  --model $model \
  --features M \
  --root_path ./data/ETT-small/ \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 0.0001 \
  --freq t \
  --d_model 512 \
  --d_ff 2048

# ETTm2
python -u run.py \
  --model $model \
  --features M \
  --root_path ./data/ETT-small/ \
  --data_path ETTm2.csv \
  --data ETTm2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --learning_rate 0.0001 \
  --freq t \
  --d_model 512 \
  --d_ff 2048
