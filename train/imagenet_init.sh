# Run as nohup this_script.sh & 

# Mount disk 
sudo mkdir -p /mnt/disks/data
sudo mount -o discard,defaults /dev/sdb /mnt/disks/data
sudo chmod a+w /mnt/disks/data

# Get zip file from bucket
cd /mnt/disks/data/imagenet_zip
gsuitl cp gs://pathway_perceiver/imagenet-object-localization-challenge.zip

# Unzip 
sudo apt install unzip 
unzip imagenet-object-localization-challenge.zip -d imagenet-object-localization-challenge
cd imagenet-object-localization-challenge
tar xvf imagenet_object_localization_patched2019.tar.gz

# Remove PNG file
cd ILSVRC/Data/CLS-LOC/train/
sudo rm n02105855/n02105855_2933.png

# Create properly formatted tarballs 
ls -1 | xargs -I '{}' tar cvf {}.tar {}
tar cvf ILSVRC2019_img_train.tar ./*.tar

cd ../test
ls > foo && tar cvf ILSVRC2019_img_test.tar -T foo  # exclude the directory `./`
cd ../val
tar cvf ILSVRC2019_img_val.tar $(ls *.JPEG)  # exclude the directory `./`
cd ../

# Move tar balls
DATASET_PATH=/mnt/disks/data/imagenet
echo ">>>> Data saving path: $DATASET_PATH"
mv train/ILSVRC2019_img_train.tar test/ILSVRC2019_img_test.tar val/ILSVRC2019_img_val.tar $DATASET_PATH

# Process tar balls with tfds
python3 <<DOC
    import os
    import tensorflow_datasets as tfds 

    base_dir = "/mnt/disks/data"
    data_dir = os.path.join(base_dir, "tensorflow_datasets")
    extract_dir = os.path.join(data_dir, "temp")
    manual_dir = os.path.join(base_dir, "imagenet")

    download_config = tfds.download.download_config(
        extract_dir=extract_dir,
        manual_dir=manual_dir,
    )

    ds = tfds.load(
        "imagenet2012:5.*.*", 
        data_dir=data_dir,
        download_and_prepare_kwargs={"download_config": download_config}
    )

DOC
