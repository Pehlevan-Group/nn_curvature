# set up parameters 
w="20"           # width of intermediate layer 
output_dim='2'  # output dimension
epochs="150000" # epochs 
save_freq="3750"
steps="5"      # steps
seed="1"        # seed 
lr="0.05"       # learning rate 
weight_decay="0"  # weight decay 
loss='CrossEntropyLoss'
data="sindata"
scanbatchsize="5"
nl="Sigmoid"
tag='sindata'

for seed in "401" "403" "404" "405" "402" "501"; do
    for w in "5" "10" "15" "20"; do
        # give arguments
        args="--w $w --output-dim $output_dim --epochs $epochs --data $data --steps $steps --seed $seed --lr $lr --weight-decay $weight_decay --loss $loss --save-freq $save_freq --nl $nl --tag $tag" 
        # train 
        # python src/run_xor.py $args --scanbatchsize $scanbatchsize --use-analytic

        # vis 
        python src/vis_xor.py $args --plot-mode "empirical" --plot-line
    done 
done