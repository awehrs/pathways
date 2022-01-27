# Constants
TPU_NAME=tpu_name
EUROPE=europe-west4-a
US=us-central1-f
V2=v2-8
V3=v3-8
DISK=pathwayperceiver/zones/europe-west4-a/disks/disk-1

# Variables
ZONE=$1
CONFIG_FILE=$2

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
--version=v2-alpha \
--data-disk source=$DISK 

# SSH into VM
# Run startup script as a here-document
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
--zone $ZONE \
<< DOC 
    # Mount disk 
    sudo mkdir -p /mnt/disks/data
    sudo mount -o discard,defaults /dev/sdb /mnt/disks/data
    sudo chmod a+w /mnt/disks/data

    # Set up imagenet
    export TFDS_DATA_DIR=mnt/disks/data/tensorflow_datasets
    
    # Upgrade pip 
    /usr/bin/python3 -m pip install --upgrade pip

    # Install jax. 
    pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

    # Clone repo into that folder 
    git clone https://github.com/awehrs/pathways

    # Install requirements
    pip install --upgrade pip 
    pip install -r pathways/requirements.txt 

    # Get the right version of ml-collections, from source
    pip uninstall -y ml-collections   
    pip install git+https://github.com/google/ml_collections 

    # Run as script 
    echo $TFDS_DATA_DIR
    PYTHONPATH=.::$PYTHONPATH python3 -m pathways.train.experiment \
    --config=$CONFIG_FILE --logtostderr

    # Save TB logs to bucket 
    gsutil cp OBJECT_LOCATION gs://pathway_perceiver/
    
    # Do other things with data

    # Exit remote session
    exit 
DOC 

# Delete VM

gcloud alpha compute tpus tpu-vm delete $TPU_NAME \
--zone=$ZONE --quiet

gcloud alpha compute tpus tpu-vm delete tpu_name \
--zone=europe-west4-a --quiet