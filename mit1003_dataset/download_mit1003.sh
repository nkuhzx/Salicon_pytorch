#download the MIT1003 dataset
wget http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip
wget http://people.csail.mit.edu/tjudd/WherePeopleLook/DATA.zip
wget http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip

unzip ALLSTIMULI.zip
unzip ALLFIXATIONMAPS.zip

#move all fixation point images to a separate folder
mkdir ALLFIXATIONPTS
mv ALLFIXATIONMAPS/*fixPts* ALLFIXATIONPTS

#remove _fixMap suffix from all filenames for convenience
rename 's/_fixMap//' ALLFIXATIONMAPS/*.jpg
