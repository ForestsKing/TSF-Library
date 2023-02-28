export CUDA_VISIBLE_DEVICES=1

model=TimesNet

# ETTh1
python -u run.py \
  --model $model \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --learning_rate 0.0001 \
  --freq h \
  --d_model 32 \
  --d_ff 64

# ETTh2
python -u run.py \
  --model $model \
  --data_path ETTh2.csv \
  --data ETTh2 \
  --learning_rate 0.0001 \
  --freq h \
  --d_model 32 \
  --d_ff 64

# ETTm1
python -u run.py \
  --model $model \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --learning_rate 0.0001 \
  --freq t \
  --d_model 32 \
  --d_ff 64

# ETTm2
python -u run.py \
  --model $model \
  --data_path ETTm2.csv \
  --data ETTm2 \
  --learning_rate 0.0001 \
  --freq t \
  --d_model 32 \
  --d_ff 64
