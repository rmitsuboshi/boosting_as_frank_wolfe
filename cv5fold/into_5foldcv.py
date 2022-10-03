import pandas as pd
import os
import sys
import tqdm
from sklearn.model_selection import KFold

###########################################################################
## Constants

PATH = "/PATH/TO/DATASET/"
OUT_PATH = "./fold_dataset"

DATASET = [
    "banana",
    "breast_cancer",
    "diabetis",
    "flare_solar",
    "german",
    "heart",
    "image",
    "ringnorm",
    "splice",
    "thyroid",
    "titanic",
    "twonorm",
    "waveform",
]


PAIR_SIZE = 20
FOLD_SIZE = 5
TEST_SIZE = 0.2

RANDOM_STATE_1 = 123
RANDOM_STATE_2 = 456
###########################################################################


if PATH == "/PATH/TO/DATASET/":
    print("The path to benchmark dataset is not specified")
    sys.exit()


# Make a new directory with name `OUT_PATH`
if not os.path.exists(PATH):
    os.mkdir(OUT_PATH)


# Split each dataset into 5-fold cross validation datasets.
for dataset in DATASET:
    print(f"[Splitting {dataset} ...]")


    df = pd.read_csv(f"{PATH}/{dataset}.csv")
    y = df["class"].astype("int64")
    x = df.drop(columns=["class"])

    for column in x.columns:
        x[column] = x[column].astype("float64")

    # Split into train test pairs
    x_train, x_test, y_train, y_test = train_test_split(\
        x, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE_1\
    )


    # Save the overall training dataset for best parameter.
    df_train = x_train.join(y_train)
    df_train.to_csv(f"{OUT_PATH}/{dataset}_train.csv", index=False)


    # Save the overall training dataset for final evaluation.
    df_test = x_test.join(y_test)
    df_test.to_csv(f"{OUT_PATH}/{dataset}_test.csv", index=False)


    # Generate an instance of `KFold`
    folds = KFold(\
        n_splits=FOLD_SIZE, shuffle=True, random_state=RANDOM_STATE_2\
    )


    for k, (train_ix, test_ix) in tqdm.tqdm(enumerate(folds.split(x_train, y=y_train))):

        # Get the indices of `k`-th train dataset
        x_fold, y_fold = x_train.iloc[train_ix, :], y_train.iloc[train_ix]

        df_fold = x_fold.join(y_fold)
        df_fold.to_csv(f"{OUT_PATH}/{dataset}_fold{k}_train.csv", index=False)


        # Get the indices of `k`-th test dataset
        x_fold, y_fold = x_train.iloc[test_ix, :], y_train.iloc[test_ix]
        df_fold = x_fold.join(y_fold)
        df_fold.to_csv(f"{OUT_PATH}/{dataset}_fold{k}_test.csv", index=False)



