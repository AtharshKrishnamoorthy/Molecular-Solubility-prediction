# Molecular Solubility Prediction Web App

## Image

[![Molecular Solubility Prediction Demo](D:\Projects\Molecular Solubility Prediction\Screenshot 2024-05-22 150537.png)(D:\Projects\Molecular Solubility Prediction\Screenshot 2024-05-22 151030.png)(D:\Projects\Molecular Solubility Prediction\Screenshot 2024-05-22 151430.png)(D:\Projects\Molecular Solubility Prediction\Screenshot 2024-05-22 151855.png)


## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

The Molecular Solubility Prediction Web App provides the following features:

1. **SMILES Input**: Enter SMILES strings of molecules to calculate their descriptors.
2. **Descriptor Calculation**: Automatically compute molecular descriptors such as LogP, Molecular Weight, Number of Rotatable Bonds, and Aromatic Proportion.
3. **Prediction**: Predict the solubility (LogS) values using a pre-trained machine learning model.
4. **Visualization**: Analyze and visualize the results with histograms and scatter plots.
5. **Custom Prediction**: Input custom descriptor values using sliders to get predictions for new compounds.
6. **Solubility Classification**: Classify predicted LogS values into solubility classes (Poorly Soluble, Moderately Soluble, Highly Soluble).

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- RDKit
- Scikit-learn
- Altair
- Pillow (PIL)
- Pickle

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/AtharshKrishnamoorthy/Molecular-Solubility-Prediction
    cd Molecular-Solubility-Prediction
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download or create the pre-trained model file (`solubility_model.pkl`) and place it in the project directory.

## Usage

To run the Molecular Solubility Prediction Web App:

1. Navigate to the project directory.
2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3. Open your web browser and go to `http://localhost:8501` (or the address provided in the terminal).

### SMILES Input

1. Enter SMILES strings in the sidebar text area.
2. Press "Enter" to input the data.

### Descriptor Calculation

1. The app will automatically compute molecular descriptors for the provided SMILES strings.

### Prediction

1. View the computed descriptors and predicted LogS values.
2. Use the interactive charts to analyze the results.

### Custom Prediction

1. Use the sliders in the sidebar to input custom descriptor values.
2. Click the "Predict" button to get predictions for the custom input.

## Model Training

The model was trained using various machine learning algorithms, including:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor

The Random Forest Regressor with 100 estimators was selected as the final model based on performance metrics.

### Training Steps:

1. **Load Dataset**: Download and load the dataset from the given URL.
2. **Preprocess Data**: Split data into training and testing sets.
3. **Train Models**: Fit various models to the training data.
4. **Evaluate Models**: Calculate and compare mean squared error for each model.
5. **Save Final Model**: Save the best-performing model using `joblib`.

## Configuration

Ensure you have the required environment setup for running the app. Modify `app.py` as needed for additional configurations.

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the original branch: `git push origin feature-branch-name`.
5. Create a pull request.

Please update tests as appropriate and adhere to the project's coding standards.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

If you have any questions, feel free to reach out:

- Project Maintainer: Atharsh K
- Email: atharshkrishnamoorthy@gmail.com
- Project Link: [GitHub Repository](https://github.com/AtharshKrishnamoorthy/Molecular-Solubility-Prediction)
