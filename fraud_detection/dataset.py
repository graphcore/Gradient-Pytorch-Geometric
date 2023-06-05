import torch
import pandas as pd
from torch_geometric.data import HeteroData, InMemoryDataset

from tqdm import tqdm

# TODO: Tidy this

def get_categorical_features(feat_df, cat_cols):
    class CategoricalEncoder:

        def __init__(self, key):
            self.key = key

        def __call__(self, df):
            categories = set(
                row[self.key] for _, row in df.iterrows())
            mapping = {cat: i for i, cat in enumerate(categories)}

            x = torch.zeros(len(df), len(mapping))
            for i, row in df.iterrows():
                x[i, mapping[row[self.key]]] = 1
            return x

    cat_encoders = [CategoricalEncoder(key) for key in cat_cols]
    cat_features = [cat_enc(feat_df) for cat_enc in tqdm(cat_encoders)]
    return torch.cat(cat_features, dim=-1)


def get_numerical_features(feat_df, num_cols):
    def nan_to_zero(val):
        return 0.0 if pd.isna(val) else val

    num_feats = [
        list(map(nan_to_zero, [row[feat] for feat in num_cols]))
        for _, row in tqdm(feat_df.iterrows())
    ]
    return torch.tensor(num_feats)


def get_edge_list(df, node_type_cols):
    # Find number of unique categories for this node type
    unique_entries = df[node_type_cols].drop_duplicates().dropna()
    # Create a map of category to value
    entry_map = {val: idx for idx, val in enumerate(unique_entries)}
    # Create edge list mapping transaction to node type
    edge_list = [[], []]

    for idx, transaction in tqdm(df.iterrows()):
        node_type_val = transaction[node_type_cols]
        # Don't create nodes for NaN values
        if pd.isna(node_type_val):
            continue
        edge_list[0].append(idx)
        edge_list[1].append(entry_map[node_type_val])
    return torch.tensor(edge_list, dtype=torch.long)


class IeeeFraudDetectionDataset(InMemoryDataset):

    url = 'https://www.kaggle.com/c/ieee-fraud-detection/data'

    non_target_node_types = [
        "card1", "card2", "card3", "card4", "card5", "card6",
        "ProductCD", "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
    excl_cols = ["TransactionID", "isFraud", "TransactionDT"]
    transaction_cat_features = [
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
        "DeviceType", "DeviceInfo", "id_12", "id_13", "id_14",
        "id_15", "id_16", "id_17", "id_18", "id_19", "id_20",
        "id_21", "id_22", "id_23", "id_24", "id_25", "id_26",
        "id_27", "id_28", "id_29", "id_30", "id_31", "id_32",
        "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"]

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])
        assert not set(self.non_target_node_types).intersection(
            set(self.excl_cols),
            set(self.transaction_cat_features)
        )

    @property
    def raw_file_names(self):
        return ['train_transaction.csv',
                'train_identity.csv',
                'test_transaction.csv',
                'test_identity.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise RuntimeError(
            f"Dataset not found. Please download {self.raw_file_names} from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    def process(self):
        train_transaction_df = pd.read_csv(self.raw_paths[0])
        train_identity_df = pd.read_csv(self.raw_paths[1])
        test_transaction_df = pd.read_csv(self.raw_paths[2])
        test_identity_df = pd.read_csv(self.raw_paths[3])

        transaction_df = pd.concat([train_transaction_df, test_transaction_df], axis=0)
        identity_df = pd.concat([train_identity_df, test_identity_df], axis=0)
        transaction_df = pd.merge(transaction_df, identity_df, on="TransactionID")
        transaction_df.sort_values("TransactionDT")

        # Remove the transactions where isFraud is NaN
        transaction_df = transaction_df[transaction_df["isFraud"].notna()]

        transaction_numeric_features = [
            column for column in transaction_df.columns
            if column not in self.non_target_node_types + self.excl_cols + self.transaction_cat_features]

        transaction_feat_df = transaction_df[transaction_numeric_features + self.transaction_cat_features].copy()
        transaction_feat_df = transaction_feat_df.fillna(0)

        print("Getting transaction categorical features...")
        transaction_cat_feats = get_categorical_features(
            transaction_feat_df,
            self.transaction_cat_features)
        print("Getting transaction numerical features...")
        transaction_num_feats = get_numerical_features(
            transaction_feat_df,
            transaction_numeric_features)
        transaction_feats = torch.cat((transaction_cat_feats, transaction_num_feats), -1)

        data = HeteroData()
        data["transaction"].num_nodes = len(transaction_df)
        data["transaction"].x = transaction_feats
        data["transaction"].y = torch.tensor(transaction_df['isFraud'], dtype=int)

        for node_type in self.non_target_node_types:
            print(f"Creating edges for {node_type} nodes...")
            edge_list = get_edge_list(transaction_df, node_type)
            data["transaction", "to", node_type].edge_index = edge_list
            data[node_type].num_nodes = edge_list[1].max() + 1
            # TODO: Shouldn't need this
            data[node_type].x = torch.zeros((edge_list[1].max() + 1, 1))
        data.validate()

        if self.pre_filter is not None:
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save([data], self.processed_paths[0])


if __name__=="__main__":
    dataset = IeeeFraudDetectionDataset(".")
    print(dataset)
    print(dataset[0])
