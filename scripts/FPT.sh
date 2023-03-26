export CUDA_VISIBLE_DEVICES=1

model=FPT
percent=100

# ETTh1
python -u run.py \
  --model $model \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --learning_rate 0.001 \
  --seq_len 336 \
  --d_model 768 \
  --stride 8 \
  --percent $percent


# ETTh2
python -u run.py \
  --model $model \
  --data_path ETTh2.csv \
  --data ETTh2 \
  --learning_rate 0.001 \
  --seq_len 336 \
  --d_model 768 \
  --stride 8 \
  --percent $percent

# ETTm1
python -u run.py \
  --model $model \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --learning_rate 0.001 \
  --seq_len 512 \
  --d_model 768 \
  --stride 16 \
  --percent $percent

# ETTm2
python -u run.py \
  --model $model \
  --data_path ETTm2.csv \
  --data ETTm2 \
  --learning_rate 0.001 \
  --seq_len 512 \
  --d_model 768 \
  --stride 16 \
  --percent $percent
