export CUDA_VISIBLE_DEVICES=1

model=NLinear

# ETTh1
python -u run.py \
  --model $model \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --learning_rate 0.001

# ETTh2
python -u run.py \
  --model $model \
  --data_path ETTh2.csv \
  --data ETTh2 \
  --learning_rate 0.001

# ETTm1
python -u run.py \
  --model $model \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --learning_rate 0.001

# ETTm2
python -u run.py \
  --model $model \
  --data_path ETTm2.csv \
  --data ETTm2 \
  --learning_rate 0.001
