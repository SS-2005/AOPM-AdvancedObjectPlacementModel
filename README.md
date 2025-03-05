# APOM-Advanced Object Placement Model in Images


## Overview:-


This project implements an advanced object placement system that automatically places an object into a background image, using a machine learning model to predict the optimal position for the object. It incorporates realistic effects like shadow creation and alpha blending for seamless integration.
It will take input from user such as object name and the image and will automatically place it into a image within seconds 


# Working:-
## To run the `APOM Code`:
Video Demonstration: 
* https://screenrec.com/share/0iCj9ugkl7
* https://drive.google.com/file/d/1adUxymj7IyYHxvMeCB7gBY3nN5Vy63d0/view?usp=drive_link
  
1) Add the dataset to your drive (To download dataset use the link `https://drive.google.com/file/d/133Wic_nSqfrIajDnnxwvGzjVti-7Y6PF/view?pli=1`)
2) Open AOPM.ipynb (link- https://colab.research.google.com/drive/1CxlxtynRcnJyfjDlZ290k2REt9sb80tL?usp=drive_link)
3) Run all the blocks sequentially.
4) Grant Permission to access your drive (The dataset is about `4.4 GB` loading it manually in colab totally impractically)
5) Enter Object name and image.
6) Result will be displayed on the screen.


   
## Features of APOM

*   **Background Removal:** Removes the background from the foreground object using the `rembg` library.
*   **Object Placement Prediction:**  Employs a convolutional neural network (CNN) to predict the ideal position (x, y coordinates and width, height) for the object within the background.
*   **Realistic Shadow Generation:** Creates and blends realistic shadows based on the object's position and size.
*   **Alpha Blending:**  Blends the foreground object with the background using alpha masks to ensure smooth edges and natural appearance.
*   **Data Augmentation:** Implements data augmentation techniques (horizontal flip, color jitter) to improve the model's generalization.


## Requirements

*   Python 3.6+
*   Libraries listed in `requirements.txt`.



## Steps:-

## Data Preparation:-

1.  **Dataset:** The code expects a dataset with the following structure:
    ```
    opa_dataset/
    ├── new_OPA/
    │   ├── background/
    │   │   ├── category1/
    │   │   │   ├── background_image1.jpg
    │   │   │   ├── background_image2.png
    │   │   │   └── ...
    │   │   ├── category2/
    │   │   │   └── ...
    │   │   └── ...
    │   ├── composite/
    │   │   ├── image1.jpg
    │   │   ├── image2.png
    │   │   └── ...
    │   ├── train_set.csv
    │   └── test_set.csv
    ```
    
    *   `background/`: Contains background images organized into categories.
      
    *   `composite/`: Contains composite images (foreground object placed on a background).
      
    *   `train_set.csv`: CSV file containing training data (image name and object position).
      
    *   `test_set.csv`: CSV file containing testing data (image name and object position).
      
    *   To download dataset use the link `https://drive.google.com/file/d/133Wic_nSqfrIajDnnxwvGzjVti-7Y6PF/view?pli=1`
   <br>Use this link to download the dataset or add its shortcut to your drive


3.  **CSV Format:** The `train_set.csv` and `test_set.csv` files should have the following columns:

    *   `img_name`:  The path to the composite image within the `composite/` directory.
    *   `position`: A string representing a list of four numbers (x, y, width, height) that define the object's bounding box in the image.  For example: `"[100, 50, 200, 150]"`.


4.  **Mount Google Drive:**  If running in Google Colab, mount your Google Drive to access the dataset:
    ```
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    Make sure to place the `opa_dataset.rar` file in your Google Drive and update the `rar_path` variable accordingly.



## Usage

1.  **Run the Code:** Execute the Python APOM script (e.g., in Google Colab).  The script will:

    *   Install necessary libraries.
    *   Load and preprocess the dataset.
    *   Train the object placement model.
    *   Prompt you to upload an image.
    *   Place the uploaded image into a randomly selected background using the trained model.
    *   Display the result.
      
2.  **Enter Object Name:** Enter the object name that you want to place

3.  **Upload Image:**  The script will use `files.upload()` (from `google.colab`) to allow you to upload a object image.

4.  **View Result:** The script will display the final image with the object placed in the background.



## Code Explanation

1.  **Dependencies:** The code starts by installing and importing necessary libraries, including `torch`, `torchvision`, `opencv-python`, `matplotlib`, `scikit-learn`, `rembg`, and `patool`.

2.  **Data Loading and Preprocessing:**
    *   Loads the training and testing datasets from CSV files.
    *   Loads background images from the specified directory.
    *   Resizes images and normalizes pixel values.
    *   Extracts object position annotations from the CSV files.

3.  **Dataset and DataLoader:**
    *   Defines a custom `PlacementDataset` class to handle image loading and transformation.
    *   Uses `DataLoader` to efficiently load data in batches during training.
    *   Applies data augmentation (random horizontal flips, color jitter) to the training data.

4.  **Model Definition:**
    *   Defines a simple CNN (`PlacementNet`) to predict the object's position (x, y, width, height) in the background.

5.  **Training:**
    *   Trains the CNN model using the training dataset and the Adam optimizer.
    *   Uses Mean Squared Error (MSE) loss to measure the difference between predicted and actual object positions.

6.  **Object Placement:**
    *   The `AdvancedObjectPlacer` class handles the object placement process.
    *   It removes the background from the foreground object using `rembg`.
    *   It predicts the object's position using the trained CNN model.
    *   It creates a realistic shadow based on the object's position and size.
    *   It blends the object with the background using alpha masks for seamless integration.

7.  **User Interaction:**
    *   Uses `files.upload()` to allow the user to upload a foreground object image.
    *   Places the uploaded object into a randomly selected background.
    *   Displays the final result using `matplotlib`.

## Model Saving

The trained model is saved as `placement_model.pth`.
For simpler understanding i have made a single code file i.e APOM


## Future Scope:

* Increase Accuracy

* Logically related background

* Reducing Overlapping and unrealistic effect

* Adjusting hyperparamets

* Integration with image prcoessing models to increase image placement effect

  

## `NOTE`: 
Current version of code does not load the entire dataset thus the image placements maybe affected, For Higher Accuracy use the entire dataset and increase the value of `NUM_EPOCH` for training and increase the `Sample_Training_Set Number`.



## Installation (Optional)

1.  Clone the repository:

    ```
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Create a virtual environment (recommended):
    ```
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

   

## Author:
* Name: Sahil Shaikh
* Email: sahil.shaikh24@aiml.sce.edu.in
