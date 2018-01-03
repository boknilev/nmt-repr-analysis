echo "Downloading multi-SNLI"
wget http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
unzip multinli_1.0

echo "Reformatting Multi-NLI dataset"
python convert_mnli.py
