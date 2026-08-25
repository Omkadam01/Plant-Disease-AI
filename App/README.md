# Application

Plant Disease AI is the Windows desktop application for grape and tomato leaf-disease classification using the repository's final EfficientNet-B0 models. It supports image and camera input, Grad-CAM visualization, English/Hindi/Marathi interface text, and optional Groq guidance.

Run `Plant-Disease-AI.exe` from this folder. Keep the repository's `Models/` and `Dataset/` folders beside `App/`; the executable reads the final model weights, YAML configuration, and class mappings from them. Python is not required to run the executable.

`app.py` is the consolidated source entry point. `Plant-Disease-AI.iss` is an Inno Setup 6 script that packages the executable with the required external model configuration, weights, and class mappings. The executable was built and tested on 64-bit Windows 11; other Windows versions were not verified.
