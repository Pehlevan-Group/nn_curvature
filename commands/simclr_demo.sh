# set up parameters 
model='34'      # width of intermediate layer 
epochs="1000"    # epochs   
steps="64"      # steps 
k="512"         # number of eigenvalues to keep
seed="501"        # seed 
lr="0.01"       # learning rate 
weight_decay="1e-4"  # weight decay 
data="cifar10"
scanbatchsize="1"  # the scan batch size 
batchsize="512"
save_epochs="0 50 200 1000" 
printfreq="1"
tag='simclr'


# give arguments
args="--model $model --epochs $epochs --data $data --seed $seed --lr $lr --weight-decay $weight_decay --save-epochs $save_epochs --tag $tag --print-freq $printfreq" 
# train 
python src/run_clr.py $args --batchsize $batchsize
python src/compute_clr.py $args --steps $steps --scanbatchsize $scanbatchsize
# vis
python src/vis_cifar.py $args --file simclr_cifar10 --steps $steps