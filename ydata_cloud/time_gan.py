"""Example using YData's time-series synthesizer."""
# ydata环境
import os
from pathlib import Path

from ydata.connectors import GCSConnector
from ydata.connectors.filetype import FileType
from ydata.metadata import Metadata
from ydata.synthesizers.timeseries.model import TimeSeriesSynthesizer
from ydata.utils.formats import read_json
from ydata.connectors import LocalConnector

def get_token(token_name: str):
    "Utility to load a token from .secrets"
    # Use relative path from file to token to be able to run regardless of the cwd()
    token_path = (
        Path(__file__)
        .absolute()
        .parent.parent.parent.parent.joinpath(".secrets", token_name)
    )
    return read_json(token_path)


if __name__ == "__main__":
    os.environ["YDATA_LICENSE_KEY"] = "ff77fa43-ede1-4ed2-9ef5-35b1d636ec85"

    TRAIN = True
    SYNTHESIZE = True

    # token = get_token("gcs_credentials.json")
    # connector = GCSConnector("bucketname", keyfile_dict=token)
    # original = connector.read_file(
    #     "gs://path-to-file/data.csv", file_type=FileType.CSV
    # )
    connector = LocalConnector()
    # Read a file from a given path
    # Dataset used: https://www.kaggle.com/datasets/uciml/adult-census-income
    original = connector.read_file("stock_data/processed_stocks.csv", file_type=FileType.CSV)

    original = original.select_columns(
        columns=[
            "timestamp",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "VOLUME",
            "PriceRange"
        ]
    )

    schema = {col: vartype.value for col, vartype in original.schema.items()}

    X = original
    dataset_attrs = {"sortbykey": "timestamp"}

    metadata = Metadata()
    m = metadata(X, dataset_attrs=dataset_attrs)

    out_path = "../test_trained_model.pkl"

    if TRAIN is True:
        synth = TimeSeriesSynthesizer()
        synth.fit(X, metadata=metadata)
        synth.save(out_path)

    if SYNTHESIZE is True:
        synth = TimeSeriesSynthesizer.load(out_path)
        sample = synth.sample(n_entities=1)
        print(sample)
        # Convert Dataset to pandas DataFrame
        sample_df = sample.to_pandas()

        # Save synthetic data to CSV file
        sample_df.to_csv("test_synth_samples.csv", index=False)

        print("Synthetic samples saved to 'test_synth_samples.csv'")