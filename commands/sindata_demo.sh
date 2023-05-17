# set up parameters 
w="250"           # width of intermediate layer 
output_dim='2'  # output dimension
epochs="30" # epochs 
save_freq="5"
steps="80"      # steps
seed="401"        # seed 
lr="0.1"       # learning rate 
weight_decay="0"  # weight decay 
loss='CrossEntropyLoss'
data="sindata"
scanbatchsize="400"
nl="Sigmoid"
tag='sindata'

# give arguments
args="--w $w --output-dim $output_dim --epochs $epochs --data $data --steps $steps --seed $seed --lr $lr --weight-decay $weight_decay --loss $loss --save-freq $save_freq --nl $nl --tag $tag" 
# train 
python src/run_xor.py $args --scanbatchsize $scanbatchsize --use-analytic

# vis 
python src/vis_xor.py $args --plot-mode "empirical" --plot-line
