import torch

class Configs:
    def __init__(self):
        self.base_path = "./all_data/"
        self.train_input_file = self.base_path + "train/radiology/semtypes.txt"
        self.train_target_file = self.base_path + "train/radiology/captions.txt"
        self.valid_input_file = self.base_path + "validation/radiology/semtypes.txt"
        self.valid_target_file = self.base_path + "validation/radiology/captions.txt"
        self.test_input_file = self.base_path + "test/radiology/semtypes.txt"
        self.test_target_file = self.base_path + "test/radiology/captions.txt"
        
        self.num_train_epochs = 5
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.portion = 0.001
        
        self.pretrained_model = "t5-base"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")