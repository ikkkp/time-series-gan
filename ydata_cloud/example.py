"""Example using YData's profiling report, tabular synthetic data generator and metrics report."""
# ydata环境
import os
from ydata.connectors import LocalConnector
from ydata.connectors.filetype import FileType
from ydata.profiling import ProfileReport
from ydata.metadata import Metadata
from ydata.synthesizers.regular.model import RegularSynthesizer
from ydata.report import SyntheticDataProfile
import pandas as pd
import numpy as np

# Definition for the fields to be anonymized before the synthesis
anonymizer_config = {
    'fnlwgt': {'type': 'regex', 'regex': r'[0-9]{6}'},
}

# Definition for the business rules
def get_education_mapping(education_num: pd.Series) -> pd.Series:
    "Maps the Education with the Education Level because it's a static relation."
    code_mapping = {
        'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4,
        '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9,
        'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12,
        'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16
    }

    # Create DataFrames from input series
    df = pd.DataFrame({'education.num': education_num})

    # Map 'Education' column using the dictionary
    # In this case if there are no matches we have decide to fill the records with "Unknown"
    df['education'] = df['education.num'].map(code_mapping).fillna('Unknown')
    return df['education']

if __name__ == "__main__":

    # Set up my YData License Key
    os.environ["YDATA_LICENSE_KEY"] = "ff77fa43-ede1-4ed2-9ef5-35b1d636ec85"

    # Example using the LocalConnector
    # Check the list of Connectors to connect diretly to your data source
    connector = LocalConnector()
    # Read a file from a given path
    # Dataset used: https://www.kaggle.com/datasets/uciml/adult-census-income
    data = connector.read_file("./stock_data.csv", file_type=FileType.CSV)
    print(data.head())

    # calculating the metadata
    metadata = Metadata(data)
    print(metadata)

    #for more details on the extracted statistics
    for item, values in metadata.summary.items():
        print('\n\033[4m'+item+'\033[0m')
        print(values)

    # Profile your data to understand what changes need to be made
    report = ProfileReport(data, title='My first Profile Report using YData')
    report.to_file('data_profiling.html') #This will save the report as a shareable HTML file

    # Choose the privacy level from
    # PRIVACY_LEVELS = ["HIGH_FIDELITY",
    #                   "BALANCED_PRIVACY_FIDELITY",
    #                   "HIGH_PRIVACY"]
    privacy_level="HIGH_FIDELITY"

    # Add your Calculated Features to the configuration
    calculated_features = [
        {
            "calculated_features": "education",
            "function": get_education_mapping,
            "calculated_from": ["education.num"],
        },
    ]

    # Instantiate a synthesizer
    synth = RegularSynthesizer()

    # fit model to the provided data
    synth.fit(data,
              metadata=metadata,
              anonymize=anonymizer_config,
              calculated_features=calculated_features,
              privacy_level=privacy_level)

    # Generate data samples by the end of the synth process
    synth_sample = synth.sample(n_samples=len(data))
    synth_metadata = Metadata(synth_sample)
    print(synth_metadata)

    # Profile your synthetic data
    # Exclude anonymized columns from the comparision
    cols = list(set(synth_sample.columns) - set(['fnlwgt']))
    report_synth = ProfileReport(synth_sample, title='My first Profile Report for Synthetic Data using YData')
    report_synth.to_file('synthetic_profiling.html') #This will save the report as a shareable HTML file

    # Compare the synthetic data with the original data
    comparison_report = report.compare(report_synth)
    comparison_report.to_file("comparison_profiling_report.html") #This will save the report as a shareable HTML file

    # Write parquet file to your destination
    connector.write_file(synth_sample.to_pandas(), path="./data.parquet", file_type=FileType.PARQUET)

    # Generate the Synthetic Data Metrics Report
    # If your dataset has a TARGET variable, you should pass it as an argument
    metrics_report = SyntheticDataProfile(
        data[cols],
        synth_sample[cols],
        metadata=metadata,
        data_types=synth.data_types)

    metrics_report.generate_report(
        output_path="./synthetic_data_metrics_report.pdf", #This will save the report as a shareable PDF file
    )