import numpy as np


class FeatureEncoder:
    # class for encoding a single categorial feature into 1-hot representation
    # The table is expected to have a HADM_ID-column

    def __init__(self, column_name, min_frequency_of_feature, max_number_features):
        self.column_name = column_name
        self.min_frequency_of_feature = min_frequency_of_feature
        self.max_number_features = max_number_features
        self.feature_mapping = None

    def fit(self, table):
        unique_element_table = table.groupby(self.column_name).count().sort_values(by='HADM_ID', ascending=False)
        # implement restriction max_number_features
        unique_element_table = unique_element_table.head(self.max_number_features)
        # implement restriction min_frequency_of_feature
        unique_element_table = unique_element_table[unique_element_table['HADM_ID'] >= self.min_frequency_of_feature]
        feature_list = unique_element_table.index.values.tolist()
        # prepare feature mapping
        feature_mapping = {}
        for i in range(0, len(feature_list)):
            feature_mapping[feature_list[i]] = i
        self.feature_mapping = feature_mapping

    def transform(self, table, hadm_id_mapping):
        # prepare feature-matrix
        feature_matrix = np.zeros((len(hadm_id_mapping), len(self.feature_mapping)), dtype=np.int8)
        for i in range(0, len(table)):
            feature = table.iloc[i][self.column_name]
            hadm_id = table.iloc[i]['HADM_ID']

            if feature in self.feature_mapping:
                row = hadm_id_mapping[hadm_id]
                col = self.feature_mapping[feature]
                feature_matrix[row, col] = 1

        return feature_matrix
