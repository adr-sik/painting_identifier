# painting_identifier

The aim of the project is to identify painting based on provided image. Utilizing feature vectors the script returns 10 closest matches.
The training set is besed on images from WikiArt database (https://archive.org/download/wikiart-dataset). These images are turned to feature vectors for compact analysis. 
Since there are issues with aspect ratios, input photos should ideally replicate the dataset's aspect ratio. Some padding is applied to address this.
data_preprocessor.py is the script used to prepare the images into .npy files. It utilizes functions of feature_extractor to apply padding and extract features using 
a pre-trained InceptionV3 model.

## Data Files

The data is split into two parts:
  details(general info like the author) - https://drive.google.com/file/d/1czwmpBQvABY86BlhSipVmNGOdn5czQjK/view?usp=drive_link
  features(used for comparison) - https://drive.google.com/file/d/1ph_JPdmw419u3icFSOypXjwcdBknC4zM/view?usp=drive_link

## Usage

To test the solution all files except data_preprocessor.py are needed.
Run main.py from the command line and provide path to an image to get results.
Type 'exit' to terminate the application.
