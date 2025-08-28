# Project installation

The project runs smoothly on WSL and Linux, but there are issues with running it purely on Windows and macOS, which would require significant changes to the code. Therefore, the installation scripts are adapted for WSL/Linux.

If you are working on Linux, simply run the “install_fixed.sh” script to properly prepare the project for launch. If you are using WSL, you must first install Node.js on Windows. The installer can be found at this link: https://nodejs.org/en/download/.


The “install_fixed.sh” script is located in the main folder of the repository. Run it with the command:
```bash
source install_fixed.sh
```

The installation may take a while, but it should run without any errors. Once the installation is complete, the project is ready to run.

# Project launch

An additional folder called test_and_scripts has been added to the repository, containing the test and scripts folders. The test folder contains a sample dataset that can be used to test the project. To do this, run the following command in the test folder(when virtual env is activated):
```bash
clip-retrieval end2end test_10000.parquet output
```

The scripts folder contains scripts that enable you to run the project on custom data. 

The main execution script is “end2end_customData.py,” which goes through the entire process of preparing embeddings, indexing, and running the application's backend and frontend. The only thing you need to set in it is the path to the folder with photos and the path to the output folder (where embeddings, indexes, etc. will be saved). This script is executed once.

When we want to run only the backend and frontend of the application (if we have already gone through the entire process in the end2end script), we need to run the “run_back.py” script after setting the path to the output folder where the indexes are located.

The “run_api_request.py” script is used to send a query to the backend. The backend address, indice_name, and query content must be set correctly.

The “run_requests_from_excel.py” script is used to send annotations contained in an Excel file as queries to the backend. This script saves the query content, response data, and the duration of each query in seconds to a .csv file. The backend address, indice_name, name of the input Excel file, and name of the output CSV file must be set correctly.

# Image Format Requirements
The “prepare_images.py” script is used to prepare photos (set appropriate formats, etc.). All you need to do is set the path to the folder and run the script.

All images must be in JPEG format.

Palette-based images (P mode) are converted to RGBA first.

Images with transparency (RGBA) have their alpha removed and a white background applied, resulting in RGB.

All other images are directly converted to RGB.

This ensures all images are RGB JPEGs without transparency.