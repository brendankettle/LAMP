
class meta():
    """Base class for metas
    """

    def __init__(self, exp_obj, config):
        self.ex = exp_obj # pass in experiment object
        self.config = config # passed in from from metas config file
        self.data_folder = self.ex.config['paths']['data_folder'] # Path()?
        return
