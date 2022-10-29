#!/bin/bash
# Run from root folder with: bash scripts/install.sh

ENV_NAME=causal

echo "Creating conda env named matcher..."
conda env remove -n $ENV_NAME
conda create -n $ENV_NAME python=3.9 pip -y -q

eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Confirm that we are in the right environment

# Assert python version
echo "Python version should be 3.9"
python --version
python -c "import sys; assert sys.version_info >= (3, 9)"


# echo "Installing dependencies..."
# TORCH="1.12.1"
# CUDA_TAG=""


# echo "Found CUDA version: $CURRENT_CUDA_VERSION"

# # If less then DESIDER2, quit
# if [ "$(echo "$CURRENT_CUDA_VERSION < $DESIRED2" | bc)" -eq 1 ]; then
#     echo "CUDA version is less than $DESIRED2. Quitting."
#     exit 1
# fi

# # check if current cuda version is greater than or equal to desired
# if [ "$CURRENT_CUDA_VERSION" \> "$DESIRED1" ]; then
#     echo "Installing torch with cuda $DESIRED1"
#     CUDA_TAG="cu113"
#     conda install pytorch torchvision torchaudio cudatoolkit=$DESIRED1 -c pytorch -y
# else
#     if [ "$CURRENT_CUDA_VERSION" \> "$DESIRED2" ]; then
#         echo "Installing torch with cuda $DESIRED1"
#         CUDA_TAG="cu102"
#         conda install pytorch torchvision torchaudio cudatoolkit=$DESIRED2 -c pytorch -y
#     else
#         echo "Error: CUDA version ${CURRENT_CUDA_VERSION} is not supported."
#         exit 1
#     fi
# fi

pip install -r requirements.txt

pip install -e .