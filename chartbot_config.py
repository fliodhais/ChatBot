# File Path

raw_data_path = "./raw_data/phase1_74Q.csv"
processed_data_path = "./clean_data/phase1_74Q_DatasetDict"
result_file_path = "./result/phase1_74Q.csv"
model_path = "./model/phase1_74Q"

# Training parameter
seed = 2046
test_size = 0.2
validation_size = 0.2

checkpoint = "bert-base-uncased"
batch_size = 32
max_num_epochs = 10
