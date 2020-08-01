wget https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/archive/master.zip
unzip master.zip
cp -a predicting-human-gaze-beyond-pixels-master/. .
rm -rf predicting-human-gaze-beyond-pixels-master
rm master.zip

unzip fixation_maps.zip -d data/
