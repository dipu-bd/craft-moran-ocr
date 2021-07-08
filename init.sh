#!bin/sh

DATASET_FOLDER="data"

verify_checksum() {
    local file=$1
    local expect_checksum=$2

    if [ ! -f "$out_file" ]; then
        return 1
    fi

    local checksum=$(md5sum $file | cut -d' ' -f1)
    if [ "$checksum" = "$expect_checksum" ]; then
        echo "$file | $checksum | verified"
        return 0
    else
        echo "$file | $checksum | not verified"
        rm -f "$file"
        return 1
    fi
}

download_google_drive_file() {
    local file_id=$1
    local checksum=$2
    local out_file=$3

    verify_checksum "$out_file" "$checksum"
    if [ $? = 0 ]; then
        return 0
    fi

    curl -L "https://drive.google.com/uc?id=$file_id&export=download" > "$out_file"
    verify_checksum "$out_file" "$checksum"

    if [ $? = 0 ]; then
        return 0
    else
        echo
        echo "$out_file > Checksum verification failed"
        return 1
    fi
}

# CREATE DATASET FOLDER
if [ ! -d "$DATASET_FOLDER" ]; then
    mkdir -v "$DATASET_FOLDER"
fi
cd "$DATASET_FOLDER"
echo

echo "--------- DOWNLOADING TRAINED DATA FOR CRAFT ------------"
download_google_drive_file \
    "1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ" \
    "2f8227d2def4037cdb3b34389dcf9ec1" \
    "craft_mlt_25k.pth"
echo

echo "---------- DOWNLOADING REFINER DATA FOR CRAFT -----------"
download_google_drive_file \
    "1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO" \
    "3d48f83540567d2a43d2f6ca4b4d9f13" \
    "craft_refiner_CTW1500.pth"
echo

echo "---------- DOWNLOADING TRAINED DATA FOR MORAN -----------"
download_google_drive_file \
    "1IDvT51MXKSseDq3X57uPjOzeSYI09zip" \
    "f1417448c934db65572f9fc261e18f09" \
    "moran_v2_demo.pth"
echo

cd ..
echo

echo "READY!"
echo
