# set up parameters 
w="2"           # width of intermediate layer 
output_dim='1'  # output dimension
epochs="0"   # epochs 
steps="40"      # steps
seed="5"        # seed 
lr="0.02"       # learning rate 
weight_decay="1e-4"  # weight decay 
loss="MSELoss"
data="XOR"
scanbatchsize="50"  # the scan batch size 

# for w in "50" "100" "250" "500" "1000"; do
for w in "2"; do
# give arguments
    args="--w $w --output-dim $output_dim --epochs $epochs --data $data --steps $steps --seed $seed --lr $lr --weight-decay $weight_decay --loss $loss --save-freq 1" 
    # train 
    python src/run_xor.py $args --scanbatchsize $scanbatchsize 

    # vis 
    python src/vis_xor.py $args --plot-mode "analytic" 
done
