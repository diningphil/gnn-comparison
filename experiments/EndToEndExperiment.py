from models.gnn_wrapper.NetWrapper import NetWrapper

from experiments.Experiment import Experiment


class EndToEndExperiment(Experiment):

    def __init__(self, model_configuration, exp_path):
        super(EndToEndExperiment, self).__init__(model_configuration, exp_path)

    def run_valid(self, dataset_getter, logger, other=None):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """

        # print(self.model_config, dataset_getter.outer_k, dataset_getter.inner_k)

        dataset_class = self.model_config.dataset  # dataset_class()

        if 'dense' in self.model_config:
            dataset = dataset_class(dense=self.model_config.dense)
        else:
            dataset = dataset_class()

        model_class = self.model_config.model
        loss_class = self.model_config.loss
        optim_class = self.model_config.optimizer
        sched_class = self.model_config.scheduler
        stopper_class = self.model_config.early_stopper
        clipping = self.model_config.gradient_clipping

        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True

        train_loader, val_loader = dataset_getter.get_train_val(dataset, self.model_config['batch_size'],
                                                                shuffle=shuffle)

        model = model_class(dim_features=dataset.dim_features, dim_target=dataset.dim_target, config=self.model_config)
        net = NetWrapper(model, loss_function=loss_class(), device=self.model_config['device'])

        optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])

        if sched_class is not None:
            scheduler = sched_class(optimizer)
        else:
            scheduler = None

        train_loss, train_acc, val_loss, val_acc, _, _, _ = net.train(train_loader=train_loader,
                                                                   max_epochs=self.model_config['classifier_epochs'],
                                                                   optimizer=optimizer, scheduler=scheduler,
                                                                   clipping=clipping,
                                                                   validation_loader=val_loader,
                                                                   early_stopping=stopper_class,
                                                                   logger=logger)
        return train_acc, val_acc

    def run_test(self, dataset_getter, logger, other=None):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR TRAINING OR EARLY STOPPING!
        :return: (training accuracy, test accuracy)
        """

        dataset_class = self.model_config.dataset  # dataset_class()

        if 'dense' in self.model_config:
            dataset = dataset_class(dense=self.model_config.dense)
        else:
            dataset = dataset_class()

        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True

        model_class = self.model_config.model
        loss_class = self.model_config.loss
        optim_class = self.model_config.optimizer
        sched_class = self.model_config.scheduler
        stopper_class = self.model_config.early_stopper
        clipping = self.model_config.gradient_clipping

        train_loader, val_loader = dataset_getter.get_train_val(dataset, self.model_config['batch_size'],
                                                                shuffle=shuffle)
        test_loader = dataset_getter.get_test(dataset, self.model_config['batch_size'], shuffle=shuffle)

        model = model_class(dim_features=dataset.dim_features, dim_target=dataset.dim_target,
                            config=self.model_config)
        net = NetWrapper(model, loss_function=loss_class(), device=self.model_config['device'])

        optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])

        if sched_class is not None:
            scheduler = sched_class(optimizer)
        else:
            scheduler = None

        train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, _ = \
            net.train(train_loader=train_loader, max_epochs=self.model_config['classifier_epochs'],
                      optimizer=optimizer, scheduler=scheduler, clipping=clipping,
                      validation_loader=val_loader, test_loader=test_loader, early_stopping=stopper_class,
                      logger=logger)

        return train_acc, test_acc
