mkdir rte
cd rte
wget http://decomp.net/wp-content/uploads/2017/11/inference_is_everything.zip
unzip inference_is_everything.zip
rm inference_is_everything.zip
cd ../
echo "About to split the data into formats for train.lua and eval.lua"
python split-data.py

echo "Downloading SNLI"
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip

echo "Reformatting SNLI dataset"
python convert_snli.py

echo "Downloading multi-SNLI"
wget http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
unzip multinli_1.0

echo "Reformatting Multi-NLI dataset"
python convert_mnli.py

echo "Downloading Compositional NLI"
mdkir compositional-rte
cd compositional-rte
svn export https://github.com/ishita-dg/ScrambleTests/trunk/testData/
