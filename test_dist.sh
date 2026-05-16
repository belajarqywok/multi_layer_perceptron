#!/bin/bash
echo "Starting MLP Master..."
./build/mlp --input-dataset=data/classification.csv --column-label=class --column-train=f1,f2,f3 --alg-type=classifier --option-std --hyperp-epochs=200 --output-onnx=dist_model.onnx --type-training=distributed --dist-role=master --dist-master-port=8085 &
MASTER_PID=$!

sleep 1

echo "Starting 3 MLP Workers..."
./build/mlp --input-dataset=data/classification.csv --column-label=class --column-train=f1,f2,f3 --alg-type=classifier --option-std --hyperp-epochs=200 --type-training=distributed --dist-role=worker --dist-master-addr=127.0.0.1 --dist-master-port=8085 &
WORKER1_PID=$!

./build/mlp --input-dataset=data/classification.csv --column-label=class --column-train=f1,f2,f3 --alg-type=classifier --option-std --hyperp-epochs=200 --type-training=distributed --dist-role=worker --dist-master-addr=127.0.0.1 --dist-master-port=8085 &
WORKER2_PID=$!

./build/mlp --input-dataset=data/classification.csv --column-label=class --column-train=f1,f2,f3 --alg-type=classifier --option-std --hyperp-epochs=200 --type-training=distributed --dist-role=worker --dist-master-addr=127.0.0.1 --dist-master-port=8085 &
WORKER3_PID=$!

echo "Waiting for workers to finish..."
wait $WORKER1_PID
wait $WORKER2_PID
wait $WORKER3_PID

echo "Workers finished. Shutting down master..."
./build/mlp --input-dataset=data/classification.csv --column-label=class --column-train=f1,f2,f3 --alg-type=classifier --option-std --hyperp-epochs=200 --type-training=distributed --dist-role=shutdown --dist-master-addr=127.0.0.1 --dist-master-port=8085

wait $MASTER_PID
echo "Distributed training completed successfully!"
