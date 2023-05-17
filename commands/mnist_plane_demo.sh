# set up parameters 
w="200"           # width of intermediate layer 
epochs="200"     # epochs 
steps="60"      # steps
seed="6"        # seed 
lr="0.01"       # learning rate 
weight_decay="1e-4"  # weight decay 
data="mnist_small"
scanbatchsize="100"  # the scan batch size 
batchsize="200"
savefreq="20"
eigvals_epochs="0 50 200"
tag='mnist'

# give arguments
args="--w $w --epochs $epochs --data $data --steps $steps --seed $seed --lr $lr --weight-decay $weight_decay --save-freq $savefreq --eigvals-epochs $eigvals_epochs --tag $tag" 
# train 
# python src/run_mnist_plane.py $args --scanbatchsize $scanbatchsize --batchsize $batchsize

# vis 
python src/vis_mnist_plane.py $args --ternary
