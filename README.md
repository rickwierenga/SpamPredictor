# Spam classifier

Train a model that predicts wether a given email is spam or not using an SVM (support vector machine).

## Installation
```shell
python3 -m pip install -r requirements.txt
```

## Running the program
Download the data:
```shell
chmod +x download_data.sh
./download_data.sh
```

Then extract the features:
```shell
chmod +x extract_features.py
./extract_features.py
```

Finally you can train the model using:
```shell
chmod +x train.py
./train.py
```

---
&copy; 2019 Rick Wierenga
