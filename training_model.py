import pandas as pd
from pipeline import pipeline
import config
import joblib

import warnings

warnings.simplefilter(action='ignore')


def run_training():
    """ Train the model"""

    # read and clean the training data
    data = pd.read_csv(config.TRAINING_DATA_FILE)

    pipeline.fit(data[config.FEATURES], data[config.TARGET],)
    joblib.dump(pipeline, config.PIPELINE_NAME, compress=('zlib',9))


if __name__ == '__main__':
    run_training()
