python -m torch.distributed.launch \
                --nproc_per_node=4 \
                --master_port=$((RANDOM + 20000)) \
                ./cifar10.py \
                --epochs=90\
                --batch_size=32\
                --repeat=1\
                --data="./datasets/cifar10"
                --log_file_name="cifar10-distributed";