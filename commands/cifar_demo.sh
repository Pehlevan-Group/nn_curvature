# set up parameters 
model='34'      # width of intermediate layer 
epochs="500"    # epochs 
steps="64"      # steps
k="512"          # number of eigenvalues to keep
seed="501"        # seed 
lr="0.01"       # learning rate 
nl="GELU"
weight_decay="1e-4"  # weight decay 
data="cifar10"
scanbatchsize="5"  # the scan batch size 
batchsize="1024"
save_epochs="0 50 200 500"
printfreq="1"
tag='cifar_gelu'

# give arguments
args="--model $model --epochs $epochs --data $data --seed $seed --lr $lr --nl $nl --weight-decay $weight_decay --save-epochs $save_epochs --tag $tag --print-freq $printfreq" 
# train 
python src/run_resnet.py $args --batchsize $batchsize

# compute geometric entities
python src/compute_resnet.py $args --steps $steps --scanbatchsize $scanbatchsize --k $k

# vis 
python src/vis_cifar.py $args --steps $steps