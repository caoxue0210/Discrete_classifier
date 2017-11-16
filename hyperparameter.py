class Hyperparameter:
    def __init__(self):
        self.unknow = "#unknow#"
        self.unknow_id = 0
        self.class_num = 5
        self.lr = 0.5
        self.epochs = 256
        self.save_dir = 'snapshot'
        self.predict = True
        self.batch_size = 10
        self.log_interval = 1
        self.test_interval = 1
        self.save_interval = 1
