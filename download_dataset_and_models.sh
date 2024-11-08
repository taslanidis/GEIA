module purge
module load 2022
module load Miniconda3/4.12.0

source activate GEIA

PROJECT_NAME="GEIA_GROUP_14"
PROJECT_DIR="$PWD"
DOWNLOAD_FILES=1

#save the weights and data in the scratch-shared folder
if [ "$DOWNLOAD_FILES" -eq 1 ]; then
	mkdir -p "/scratch-shared/${PROJECT_NAME}"/
	cd "/scratch-shared/${PROJECT_NAME}/"

	wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1oIo8P0Y8X9DTeEfOA1WUKq8Uix9a_Pte" -O data.zip
	unzip data.zip
	rm -rf *data.zip

	cd data

	python ${PROJECT_DIR}/download_dataset.py /scratch-shared/${PROJECT_NAME}/data

	mkdir "/scratch-shared/${PROJECT_NAME}/model_weights"
	cd "/scratch-shared/${PROJECT_NAME}/model_weights"

	python ${PROJECT_DIR}/download_models.py /scratch-shared/${PROJECT_NAME}/model_weights
fi



#symbolic link
cd $PROJECT_DIR
ln -s "/scratch-shared/${PROJECT_NAME}/data"
ln -s "/scratch-shared/${PROJECT_NAME}/model_weights"
