# Instructions to run B-CQWGAN-Evol on the DSRI infrastructure
Instruction to run the script on Maastricht University [Data Science Research Infrastructure](https://dsri.maastrichtuniversity.nl/docs/
)'s servers.  

## Using a set-up script

### For CPU (via vscode-gan-evolutionary pod)

1. Open the terminal

1. Create the virtual environment and activate it. Also deactivate the conda enviromnent 
that you are automatically in at start-up:
    1. `conda deactivate`
    1. `python -m venv venv1`
    1. `source venv1/bin/activate`
    

### For GPU (via jupyterlab-gpu-torch-bianca pod)
1. Open the terminal
1. Create and activate virtual environment
    1. `apt-get update`
    1. `apt install python3.10-venv` 
    1. `python3.10 -m venv venv1`
    1. `source venv1/bin/activate`


## Install requirements

- Install requirements
- `pip install -r requirements.txt`
  
May be necessary to
- `pip install pennylane`
- `pip install qiskit`

## 
