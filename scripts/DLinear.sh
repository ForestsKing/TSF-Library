export CUDA_VISIBLE_DEVICES=1

model=DLinear

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
  --learning_rate 0.001 \
  --freq h

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
  --learning_rate 0.001 \
  --freq h

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
  --learning_rate 0.001 \
  --freq t

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
  --learning_rate 0.001 \
  --freq t