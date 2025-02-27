using CNN_interface.Model;
using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using CNN_interface.ViewModel.MenuCommands;

namespace CNN_interface.ViewModel
{
    public class MainViewModel: INotifyPropertyChanged
    {
        private CnnModel _model;
        private string _imagePath;
        private string _prediction;

        public string ImagePath
        {
            get => _imagePath;
            set
            {
                _imagePath = value;
                OnPropertyChanged(nameof(ImagePath));
            }
        }

        public string Prediction
        {
            get => _prediction;
            set
            {
                _prediction = value;
                OnPropertyChanged(nameof(Prediction));
            }
        }

        public ICommand LoadImageCommand { get; }
        public ICommand RunCnnCommand { get; }

        public MainViewModel()
        {
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "mobilenetv2_cifar10.onnx");
            _model = new CnnModel(modelPath);
            LoadImageCommand = new RelayCommand(LoadImage);
            RunCnnCommand = new RelayCommand(RunCnn);
        }

        private void LoadImage()
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Image Files (*.png;*.jpg)|*.png;*.jpg"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                ImagePath = openFileDialog.FileName;
            }
        }

        private readonly string[] _classLabels = {
            "Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"
        };

        private void RunCnn()
        {
            if (string.IsNullOrEmpty(ImagePath)) return;

            // Convert image to appropriate format
            float[] input = PreprocessImage(ImagePath);

            // Run prediction
            int predictedClassIndex = _model.Predict(input);
            string predictedClass = _classLabels[predictedClassIndex];
            Prediction = $"Predicted Class: {predictedClass}";
        }


        private float[] PreprocessImage(string imagePath)
        {
            using var bitmap = new Bitmap(imagePath);
            using var resized = new Bitmap(bitmap, new Size(32, 32)); 
            float[] input = new float[3 * 32 * 32]; 

            for (int y = 0; y < 32; y++)
            {
                for (int x = 0; x < 32; x++)
                {
                    var pixel = resized.GetPixel(x, y);

                    // Normalizeaza fiecare canal intre [-1, 1]
                    float red = (pixel.R / 255f - 0.5f) / 0.5f;
                    float green = (pixel.G / 255f - 0.5f) / 0.5f;
                    float blue = (pixel.B / 255f - 0.5f) / 0.5f;

                    int index = y * 32 + x;
                    input[index] = red;         // Canalul R
                    input[index + 32 * 32] = green;  // Canalul G
                    input[index + 2 * 32 * 32] = blue;  // Canalul B
                }
            }

            return input;
        }


        public event PropertyChangedEventHandler PropertyChanged;

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
