# Constants
TPU_NAME=tpu_name
EUROPE=europe-west4-a
US=us-central1-f
V2=v2-8
V3=v3-8
CONFIG_FILE=perceiver/train/experiment.py

# Variables
ZONE=$1

if [ $ZONE = europe ]; then
    ACCELERATOR=$V3
    ZONE=$EUROPE
elif [ $ZONE = us ]; then 
    ACCELERATOR=$V2
    ZONE=$US
else 
    echo "Specify region"
    exit 125
fi

# Create TPU VM
gcloud alpha compute tpus tpu-vm create $TPU_NAME \
--zone $ZONE \
--accelerator-type $ACCELERATOR \
--version=v2-alpha

# SSH into VM
# Run startup script as a here-document
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
--zone $ZONE \
<< DOC 
    # Install jax. 
    pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

    # Create folder 
    mkdir perceiver 

    # Clone repo into that folder 
    git clone https://github.com/awehrs/pathways perceiver

    # Install requirements
    pip install --upgrade pip 
    pip install -r perceiver/requirements.txt 

    # Get the right version of ml-collections
    pip uninstall -y ml-collections   
    pip install git+https://github.com/google/ml_collections 

    # Run as script 
    python3 -m pathways.train.experiment \
    --config=pathways/train/experiment/py --logtostderr

    # Save TB logs to ???

    # Do other things with data

    # Exit remote session
    exit 
DOC 

# Delete VM

gcloud alpha compute tpus tpu-vm delete $TPU_NAME \
--zone=$ZONE --quiet