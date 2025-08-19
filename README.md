# LoRA Fine-tuning with MLflow

A comprehensive, beginner-friendly tutorial for fine-tuning language models using LoRA (Low-Rank Adaptation) with complete MLflow experiment tracking and model management.

## 🎯 What You'll Learn

- **LoRA Fine-tuning**: Efficient fine-tuning with minimal parameters
- **MLflow Integration**: Complete experiment tracking and model registry
- **Model Evaluation**: Systematic testing and performance measurement
- **Best Practices**: Production-ready ML workflows

## 🚀 Quick Start

### Prerequisites
```bash
# Clone the repository
git clone <your-repo-url>
cd mlflow-3.0_run-

# Install dependencies
pip install -r requirements.txt
```

### Setup Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key (optional, only for judge evaluation)
# OPENAI_API_KEY=your_key_here
```

### Run the Tutorial
```bash
# Start Jupyter notebook
jupyter notebook

# Open lora_finetuning.ipynb
# Follow the step-by-step guide!
```

### View MLflow Results
```bash
# Start MLflow UI (in another terminal)
mlflow ui

# Visit: http://localhost:5000
```

## 📚 Tutorial Structure

The Jupyter notebook is organized into clear, easy-to-follow sections:

### **1. Setup and Imports** 
- Install and import required libraries
- Check system compatibility (CPU/GPU/MPS)

### **2. Configuration**
- Model selection (DialoGPT-small for demo)
- Training parameters (batch size, learning rate, etc.)
- LoRA settings (rank, alpha, target modules)

### **3. MLflow Setup**
- Initialize experiment tracking
- Configure logging and metrics

### **4. Dataset Creation**
- Create instruction-following dataset
- Format for language model training
- 10 sample Q&A pairs included

### **5. Model Loading**
- Load base model and tokenizer
- Apply LoRA configuration
- Display trainable parameters (very few!)

### **6. Data Preparation**
- Tokenize dataset for training
- Set up data collators

### **7. Training Configuration**
- Configure training arguments
- Set up Trainer from Hugging Face

### **8. Model Training**
- Complete training with MLflow logging
- Track loss, time, and performance metrics
- Save model and tokenizer

### **9. Model Evaluation**
- Load fine-tuned model
- Test on sample questions
- Generate responses and measure quality

### **10. MLflow Logging**
- Log evaluation results
- Save metrics and artifacts
- Track model performance

### **11. Model Registry** (Optional)
- Register model for version control
- Prepare for deployment

### **12. Summary**
- Review accomplishments
- Suggest next steps
- Learning outcomes

## 🔧 Key Features

### **Efficient Training**
- LoRA fine-tuning uses <1% of original parameters
- Works on CPU, MPS (Apple Silicon), or CUDA
- Fast training (3-5 minutes for demo)

### **Complete MLflow Integration**
- Local experiment tracking (no external services needed)
- Automatic parameter and metric logging
- Model versioning and registry
- Artifact management
- Visual progress bars with tqdm

### **Beginner-Friendly**
- Step-by-step explanations
- Clear code comments
- Visual progress indicators
- Error handling and tips

### **Production-Ready**
- Best practices for model training
- Evaluation and testing workflows
- Model registry for deployment
- Comprehensive logging

## 📊 What Gets Tracked in MLflow

### **Training Metrics**
- Training loss progression
- Training time and speed
- Model parameters and configuration
- Device and system information

### **Evaluation Results**
- Response quality metrics
- Average response length
- Success rates
- Sample Q&A pairs

### **Model Artifacts**
- Trained LoRA adapters
- Tokenizer configurations
- Model checkpoints
- Evaluation results (CSV)

### **Configuration**
- All hyperparameters
- Dataset information
- Model architecture details
- Training environment

## 🎓 Learning Outcomes

After completing this tutorial, you'll understand:

### **LoRA Fine-tuning**
- How LoRA reduces training parameters by 99%
- When to use LoRA vs full fine-tuning
- Configuring rank, alpha, and target modules
- Training instruction-following models

### **MLflow Mastery**
- Setting up experiments and runs
- Logging parameters, metrics, and artifacts
- Using the Model Registry for versioning
- Viewing and comparing results in UI

### **Best Practices**
- Organizing ML experiments
- Evaluating model performance
- Managing model lifecycle
- Preparing models for production

## 🛠️ Customization

The notebook is designed to be easily customizable:

### **Change the Model**
```python
config = {
    "model_name": "microsoft/DialoGPT-medium",  # Try different sizes
    # ... other settings
}
```

### **Adjust Training Parameters**
```python
config = {
    "batch_size": 4,        # Increase if you have more memory
    "num_epochs": 5,        # Train longer for better results
    "learning_rate": 1e-4,  # Experiment with different rates
    # ... other settings
}
```

### **Modify LoRA Configuration**
```python
config = {
    "lora_r": 16,           # Higher rank = more parameters
    "lora_alpha": 32,       # Scaling parameter
    "lora_dropout": 0.1,    # Regularization
    # ... other settings
}
```

### **Expand the Dataset**
Add more instruction-response pairs in the `create_training_dataset()` function.

## 🔧 Requirements

### **System Requirements**
- Python 3.8+
- 4GB+ RAM recommended
- GPU optional (works on CPU/MPS)

### **Key Dependencies**
- `torch>=2.0.0` - PyTorch for deep learning
- `transformers>=4.36.0` - Hugging Face models
- `peft>=0.7.0` - Parameter-efficient fine-tuning
- `mlflow>=2.10.0` - Experiment tracking
- `datasets>=2.14.0` - Dataset management
- `jupyter>=1.0.0` - Notebook environment
- `tqdm>=4.64.0` - Progress visualization

## 🚨 Troubleshooting

### **Memory Issues**
- Reduce `batch_size` to 1
- Use smaller model (DialoGPT-small)
- Reduce `max_length` parameter

### **Installation Issues**
```bash
# If you get dependency conflicts
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### **MLflow UI Issues**
```bash
# If MLflow UI won't start
pkill -f mlflow
mlflow ui --port 5001
```

### **Model Loading Issues**
- Check internet connection for model download
- Ensure sufficient disk space
- Try clearing Hugging Face cache: `~/.cache/huggingface/`

## 🎉 Next Steps

After completing the tutorial:

1. **🔬 Experiment**: Try different models and parameters
2. **📚 Expand Dataset**: Add more training examples
3. **🎯 Evaluate**: Compare different LoRA configurations
4. **🚀 Deploy**: Use MLflow Model Registry for deployment
5. **📊 Monitor**: Track model performance over time

## 📚 Additional Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Original research
- [MLflow Documentation](https://mlflow.org/docs/latest/) - Complete guide
- [Hugging Face PEFT](https://github.com/huggingface/peft) - LoRA implementation
- [Transformers Documentation](https://huggingface.co/docs/transformers) - Model library

## 🤝 Contributing

This is an educational project! Feel free to:
- Add more example datasets
- Try different model architectures
- Improve evaluation metrics
- Enhance documentation

---

**🎯 Ready to start?** Open `lora_finetuning.ipynb` and begin your fine-tuning journey!

**📊 Questions?** Check the MLflow UI at `http://localhost:5000` for detailed results.