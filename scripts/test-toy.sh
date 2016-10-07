#!bin/bash

TRAIN_FILE="data/Toy/toy-train.txt"
VALIDATION_FILE="data/Toy/toy-test.txt"
TEST_FILE="data/Toy/toy-test.txt"
ENTITY_DICTIONARY="data/Toy/entities.dic"
RELATION_DICTIONARY="data/Toy/relations.dic"
MODEL_PATH="models/toy-distmult.model"
ALGORITHM="distmult"
PREDICTION_FILE="data/temporary.txt"

THEANO_FLAGS='floatX=float32,warn_float64=raise,optimizer_including=local_remove_all_assert' python code/experts/predict.py --train_data $TRAIN_FILE --validation_data $VALIDATION_FILE --test_data $TEST_FILE --entities $ENTITY_DICTIONARY --relations $RELATION_DICTIONARY --model_path $MODEL_PATH --algorithm $ALGORITHM --prediction_file $PREDICTION_FILE

python code/evaluation/report.py --filepath $PREDICTION_FILE