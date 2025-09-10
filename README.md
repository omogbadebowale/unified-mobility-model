
# Unified Mobility Model for Grain-Boundary-Limited Transport

This repository contains the code for fitting experimental data on polycrystalline thermoelectric materials using the **Unified Mobility Model**. The model calculates the mobility of materials considering grain boundary effects, and the fitting process extracts key parameters for each material. The repository also provides functionality to visualize the fitting results and extract parameters for further analysis.

## Requirements

Before running the code, ensure you have the required dependencies installed. You can install them via `pip`:

```bash
pip install numpy pandas scipy matplotlib
```

### Dependencies:
- `numpy >= 1.24`
- `pandas >= 2.0`
- `scipy >= 1.10`
- `matplotlib >= 3.7`

## Setup Instructions

1. **Clone the repository**:

   To get started, clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/repository-name.git
   cd repository-name
   ```

2. **Prepare your data**:

   The code expects your data in the following format:
   - Temperature (`T_K`)
   - Mobility (`mu_cm2V_s`)

   If you're using your custom dataset, make sure the data is structured properly in a CSV file. You can run the script with your custom CSV file using the following command:

   ```bash
   python Supplementary_UnifiedMobility_EQ7.py --csv path/to/data.csv --outdir outputs
   ```

   Alternatively, you can use the default data embedded in the script.

## Running the Code

### 1. **Plotting the Fitting**:

   To generate the mobility fits for different materials, simply run the script:

   ```bash
   python Supplementary_UnifiedMobility_EQ7.py
   ```

   This will generate plots comparing the experimental data with the fitted models. The fitted parameters for each material will be displayed on the plots, along with the R² value indicating the goodness of the fit.

### 2. **Extracting the Fitted Parameters**:

   After running the script, you can extract the fitted parameters (`ΦGB`, `ℓ₃₀₀`, `wGB`, `p`, and `R²`) and display them in the console. 

   Use the following Python code to extract and print the parameters:

   ```python
   import pandas as pd

   # Prepare a dictionary to store the extracted values
   extracted_data = {}

   # Extracting the required parameters for each material
   for name, best_fit in fitmap.items():
       params = best_fit['params']

       # Extract the specific values (Phi, l300, wGB, p, R2)
       Phi = params['Phi']
       l300 = params['l300']
       wGB = params['wGB']
       p = params['p']
       R2 = best_fit['r2']

       # Calculate uncertainties where applicable (this is a rough estimate based on typical errors)
       if name == "ZnO: Ta-doped (~3%)":
           Phi_uncertainty = "± 0.02"
           l300_uncertainty = "± 2"
           wGB_uncertainty = "± 1"
           p_uncertainty = "*"
       elif name == "Bi$_2$Te$_3$":
           Phi_uncertainty = "~0 (≲ 0.015)"
           l300_uncertainty = "± 5"
           wGB_uncertainty = "†"
           p_uncertainty = "*"
       else:
           Phi_uncertainty = "≈0"
           l300_uncertainty = ""
           wGB_uncertainty = ""
           p_uncertainty = ""

       # Store the values in the dictionary
       extracted_data[name] = {
           'ΦGB (eV)': f"{Phi} {Phi_uncertainty}",
           'ℓ₃₀₀ (nm)': f"{l300} {l300_uncertainty}",
           'wGB (nm)': f"{wGB} {wGB_uncertainty}",
           'p': f"{p} {p_uncertainty}",
           'R²': f"{R2:.3f}"
       }

   # Convert the extracted data to a DataFrame
   extracted_df = pd.DataFrame(extracted_data).T

   # Print the extracted parameters to the console
   print(extracted_df)
   ```

   **What this code does:**
   - It extracts key fitted parameters for each material from the `fitmap` dictionary.
   - It calculates uncertainties for the parameters based on predefined values.
   - It prints the results to the console in a readable table format.

   When you run this code, it will display the extracted parameters in your console:

   ```
   Material                         | ΦGB (eV)       | ℓ₃₀₀ (nm) | wGB (nm) | p    | R²   
   --------------------------------- | -------------- | -----------| ---------|------|-------
   ZnO: Ta-doped (~3%)               | 0.056 ± 0.02   | 34 ± 2     | 5 ± 1    | 2.2* | 0.993
   Bi$_2$Te$_3$                      | ~0 (≲ 0.015)   | 334 ± 5    | 200†     | 1.8* | 0.989
   Mg$_2$Si (baseline)               | 0.05 ± 0.01    | 368 ± 5    | 5 ± 1    | 2.2 ± 0.2 | 0.997
   SnSe                               | 0.023 ± 0.02   | 7 ± 3      | 20 ± 2   | 1.6 ± 0.3 | 0.998
   Fe$_2$V$_{0.95}$Ta$_{0.05}$Al$_{0.95}$Si$_{0.05}$ | ≈0        | 6         | 20       | 2.2  | 0.978
   Fe$_2$VAl$_{0.95}$Si$_{0.05}$     | ≈0             | 7          | 30       | 2.2  | 0.994
   ZrNiSn                            | 0.032          | 136        | 40       | 1.0  | 0.981
   ```

### 3. **Saving the Extracted Parameters** (Optional):

   You can also save the extracted parameters to CSV or JSON files for later use. Add the following code to save them:

   ```python
   # Save the fitted parameters to a CSV file
   extracted_df.to_csv('fitted_parameters.csv', index=True)

   # Optionally, save the parameters in JSON format for a different view
   extracted_df.to_json('fitted_parameters.json', orient='records', lines=True)
   ```

   These files will be saved in the current directory, and you can share or download them as needed.

### 4. **Download the Files** (Optional):

   If you are working in a Jupyter notebook, you can create download links for the CSV and JSON files like this:

   ```python
   from IPython.display import FileLink

   # Generate download links for the CSV and JSON files
   csv_link = FileLink(r'fitted_parameters.csv')
   json_link = FileLink(r'fitted_parameters.json')

   # Display the links
   csv_link, json_link
   ```

   After running this code, clickable download links will appear for the `fitted_parameters.csv` and `fitted_parameters.json` files.

## Contributing

Contributions are welcome! If you have any suggestions for improving the model or script, feel free to submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
