# set up parameters 
w="2000"           # width of intermediate layer 
epochs="30"     # epochs 
steps="10"      # steps
k="784"          # number of eigenvalues to keep
seed="5"        # seed 
lr="0.01"       # learning rate 
weight_decay="1e-4"  # weight decay 
data="mnist"
scanbatchsize="2"  # the scan batch size 
batchsize="1000"
savefreq="5"
printfreq="1"
tag='mnist'

# give arguments
args="--w $w --epochs $epochs --data $data --steps $steps --seed $seed --lr $lr --weight-decay $weight_decay --save-freq $savefreq --tag $tag --print-freq $printfreq" 
# train 
python src/run_mnist.py $args --scanbatchsize $scanbatchsize --batchsize $batchsize --k $k

# vis 
python src/vis_mnist.py $args
