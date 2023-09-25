# ConGraCNet
## Construction Grammar Conceptual Networks
A method to extract conceptual networks from a corpus based on the syntactic-semantic constructions

Emocnet

# Instructions:
## Clone the Repository:

First, you'll need to clone the repository to your local machine using git.

git clone https://github.com/bperak/ConGraCNet.git

This will create a directory named EmoCNet in your current location. Navigate to that directory:

cd EmoCNet

## Create a Virtual Environment (Recommended):

It's a good practice to create a virtual environment for Python projects to avoid potential conflicts between package versions. Here's how to do it:

If you're using the standard Python environment:

python -m venv emocnet_env

And activate it:

On Windows:
.\emocnet_env\Scripts\activate

On macOS and Linux:
source emocnet_env/bin/activate

If you're using Anaconda:
conda create --name emocnet_env python=3.11
conda activate emocnet_env

## Install Dependencies:

Now, you'll want to install the required packages from the requirements.txt file:

pip install -r requirements.txt

## Run the Streamlit App:

Once the dependencies are installed, you can run the Streamlit app:

streamlit run cgcnStream_0_3_6_withSBBLabel.py

This command should start the Streamlit server and open a browser window with the app. If it doesn't automatically open, the console will provide a local URL (typically http://localhost:8501/) that you can manually enter into your browser to access the app.

