import * as ort from 'onnxruntime-node';

async function main() {
    if (process.argv.length < 3) {
        console.error("Usage: npx ts-node inference.ts <path_to_model.onnx>");
        process.exit(1);
    }

    const modelPath = process.argv[2];
    console.log(`Loading model from ${modelPath}...`);

    try {
        // Create an ONNX inference session
        const session = await ort.InferenceSession.create(modelPath);

        // Prepare the input tensor (Dummy Tensor: [1.0, 2.0, 3.0])
        const inputData = Float32Array.from([1.0, 2.0, 3.0]);
        const inputTensor = new ort.Tensor('float32', inputData, [1, 3]);

        // Get the expected input name from the session
        const inputName = session.inputNames[0];
        console.log(`Expected Input Name: ${inputName}`);
        console.log(`Input Features: [1.0, 2.0, 3.0]`);

        // Prepare the feeds
        const feeds: Record<string, ort.Tensor> = {};
        feeds[inputName] = inputTensor;

        // Run the inference session
        const results = await session.run(feeds);

        // Get the output
        const outputName = session.outputNames[0];
        const outputTensor = results[outputName];

        console.log(`Output: [${outputTensor.data[0]}]`);
    } catch (e) {
        console.error(`Failed to infer model: ${e}`);
    }
}

main();
