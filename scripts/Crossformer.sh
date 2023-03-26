export CUDA_VISIBLE_DEVICES=1

model=Crossformer
seq_len=672
seg_len=12

# ETTh1
python -u run.py \
  --model $model \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --seq_len $seq_len \
  --seg_len $seg_len \
  --factor 10 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 3 \
  --dropout 0.2 \
  --learning_rate 0.0001

# ETTh2
python -u run.py \
  --model $model \
  --data_path ETTh2.csv \
  --data ETTh2 \
  --seq_len $seq_len \
  --seg_len $seg_len \
  --factor 10 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 3 \
  --dropout 0.2 \
  --learning_rate 0.0001

# ETTm1
python -u run.py \
  --model $model \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --seq_len $seq_len \
  --seg_len $seg_len \
  --factor 10 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 3 \
  --dropout 0.2 \
  --learning_rate 0.0001

# ETTm2
python -u run.py \
  --model $model \
  --data_path ETTm2.csv \
  --data ETTm2 \
  --seg_len $seg_len \
  --factor 10 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 3 \
  --dropout 0.2 \
  --learning_rate 0.0001
