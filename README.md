# coding-ai
My try to make an AI that code an app from it's description.

## Installation

```bash
pip3 install -r requirements.txt
```

## Download dataset
```
mkdir ./tmp ./data
curl -o ./tmp/py150_files.tar.gz https://files.sri.inf.ethz.ch/data/py150_files.tar.gz
tar -xzf ./tmp/py150_files.tar.gz -C tmp
tar -xzf ./tmp/data.tar.gz
rm -rf ./tmp
```
