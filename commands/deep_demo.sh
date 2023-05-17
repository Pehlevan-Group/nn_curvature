# set up parameters 
ws="8 8 8"           # width of intermediate layer 
output_dim='2'  # output dimension
epochs="10000" # epochs 
save_freq="200"
steps="40"      # steps
seed="401"        # seed 
lr="0.1"       # learning rate 
weight_decay="0"  # weight decay 
loss='CrossEntropyLoss'
data="sindata"
scanbatchsize="100"
nl="Sigmoid"
tag='exp'

# give arguments
args="--ws $ws --output-dim $output_dim --epochs $epochs --data $data --steps $steps --seed $seed --lr $lr --weight-decay $weight_decay --loss $loss --save-freq $save_freq --nl $nl --tag $tag" 
# train 
# python src/run_deep.py $args --scanbatchsize $scanbatchsize

# vis 
python src/vis_deep.py $args --plot-mode "empirical" --plot-line
