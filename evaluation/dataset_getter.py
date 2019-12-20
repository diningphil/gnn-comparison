

class DatasetGetter:

    def __init__(self, outer_k=None, inner_k=None):
        self.outer_k = outer_k
        self.inner_k = inner_k

    def set_inner_k(self, k):
        self.inner_k = k

    def get_train_val(self, dataset, batch_size, shuffle=True):
        return dataset.get_model_selection_fold(self.outer_k, self.inner_k, batch_size, shuffle)

    def get_test(self, dataset, batch_size, shuffle=True):
        return dataset.get_test_fold(self.outer_k, batch_size, shuffle)