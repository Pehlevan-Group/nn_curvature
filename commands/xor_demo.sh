# set up parameters 
w="250"           # width of intermediate layer 
output_dim='2'  # output dimension
epochs="2000"   # epochs 
steps="40"      # steps
seed="400"        # seed 
lr="0.02"       # learning rate 
weight_decay="1e-4"  # weight decay 
loss="CrossEntropyLoss"
data="XOR"
scanbatchsize="100"  # the scan batch size 
nl="Sigmoid"
tag='exp'

# give arguments
args="--w $w --output-dim $output_dim --epochs $epochs --data $data --steps $steps --seed $seed --lr $lr --weight-decay $weight_decay --loss $loss --nl $nl" 
# train 
# python src/run_xor.py $args --scanbatchsize $scanbatchsize --use-analytic

# vis 
python src/vis_xor.py $args --plot-mode "empirical"
