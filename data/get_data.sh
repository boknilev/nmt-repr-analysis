mkdir rte
cd rte
wget http://decomp.net/wp-content/uploads/2017/11/inference_is_everything.zip
unzip inference_is_everything.zip
rm inference_is_everything.zip
cd ../
echo "About to split the data into formats for train.lua and eval.lua"
python split-data.py
