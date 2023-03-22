
MIN = 60 #sec

class Parameters:
    def __init__(self):
        self.time_minutes = 300
        self.time_period = self.time_minutes * MIN  # 2 hours
        self.nb_new_inputs = 1000 # number of newly generated inputs
