module.exports = {
  apps : [
    {
      name: "mlp-master",
      script: "./build/mlp",
      args: "--input-dataset=data/classification.csv --column-label=class --column-train=f1,f2,f3 --alg-type=classifier --option-std --hyperp-epochs=200 --output-onnx=dist_model.onnx --type-training=distributed --dist-role=master --dist-master-port=8085",
      cwd: ".",
      autorestart: false
    },
    {
      name: "mlp-worker",
      script: "./build/mlp",
      args: "--input-dataset=data/classification.csv --column-label=class --column-train=f1,f2,f3 --alg-type=classifier --option-std --hyperp-epochs=200 --type-training=distributed --dist-role=worker --dist-master-addr=127.0.0.1 --dist-master-port=8085",
      cwd: ".",
      autorestart: false,
      instances: 3, // Spawn 3 identical workers
      exec_mode: "fork"
    }
  ]
};
