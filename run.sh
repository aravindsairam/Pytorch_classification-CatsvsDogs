export MODEL="resnet34"

export MODEL_MEAN="(0.485, 0.456, 0.406)"
export MODEL_STD="(0.229, 0.224, 0.225)"
export IMG_HEIGHT=224
export IMG_WIDTH=224
export BATCH_SIZE=64

export EPOCHS=10
export LEARNING_RATE=0.001

export TRAIN_FOLDS="(0, 1, 2 ,3)"
export VAL_FOLDS="(4,)"
python3 train.py

export TRAIN_FOLDS="(1, 2 ,3, 4)"
export VAL_FOLDS="(0,)"
python3 train.py

export TRAIN_FOLDS="(2, 3, 4 ,0)"
export VAL_FOLDS="(1,)"
python3 train.py

export TRAIN_FOLDS="(3, 4, 0 ,1)"
export VAL_FOLDS="(2,]"
python3 train.py

export TRAIN_FOLDS="(4, 0,1 ,2)"
export VAL_FOLDS="(3,)"
python3 train.py

python3 test.py