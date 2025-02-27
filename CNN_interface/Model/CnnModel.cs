using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CNN_interface.Model
{
    public class CnnModel
    {
        // Responsible for loading and running an ONNX model.
        private InferenceSession _session;

        // Constructor that takes the path to the ONNX model file.
        public CnnModel(string modelPath)
        {
            _session = new InferenceSession(modelPath);
        }

        // Method that takes an input tensor and returns the predicted label.
        public int Predict(float[] input)
        {
            // Create a tensor from the input array.
            var inputTensor = new DenseTensor<float>(input, new[] { 1, 3, 32, 32 });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            // Run the model and get the output tensor.
            using var results = _session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();

            // Return the index of the maximum value (predicted label)
            return Array.IndexOf(output, output.Max());
        }
    }
}
