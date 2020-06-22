from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

import preprocessors as pp
import config


pipeline = Pipeline(
        [
            ('selector', pp.DataFrameSelector(attribute_names=config.FEATURES)),
            ('log_transformer', pp.LogTransformer(variables=config.NUMERICAL_LOG)),
            ('cat_imputer', pp.CategoricalImputer(variables=config.CATEGORICAL_WITH_NA)),
            ('rare_label_encoder', pp.RareLabelCategoricalEncoder(tol=0.01, variables=config.CATEGORICAL_VARS)),
            ('categorical_encoder', pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
            ("Random_Forest", RandomForestRegressor(n_estimators=50, min_samples_split=4, random_state=101)),
        ]
    )