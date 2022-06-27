def _after_task(self, taski, inc_dataset):
    network = deepcopy(self._parallel_network)
    network.eval()
    self._ex.logger.info("save model")
    if self._cfg["save_ckpt"] and taski >= self._cfg["start_task"]:
        save_path = os.path.join(os.getcwd(), "ckpts")
        torch.save(network.cpu().state_dict(), "{}/step{}.ckpt".format(save_path, self._task))

    if (self._cfg["decouple"]['enable'] and taski > 0):
        if self._cfg["decouple"]["fullset"]:
            train_loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="train")
        else:
            train_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                   inc_dataset.targets_inc,
                                                   mode="balanced_train")

        # finetuning
        self._parallel_network.module.classifier.reset_parameters()
        finetune_last_layer(self._ex.logger,
                            self._parallel_network,
                            train_loader,
                            self._n_classes,
                            nepoch=self._decouple["epochs"],
                            lr=self._decouple["lr"],
                            scheduling=self._decouple["scheduling"],
                            lr_decay=self._decouple["lr_decay"],
                            weight_decay=self._decouple["weight_decay"],
                            loss_type="ce",
                            temperature=self._decouple["temperature"])
        network = deepcopy(self._parallel_network)
        if self._cfg["save_ckpt"]:
            save_path = os.path.join(os.getcwd(), "ckpts")
            torch.save(network.cpu().state_dict(), "{}/decouple_step{}.ckpt".format(save_path, self._task))

    if self._cfg["postprocessor"]["enable"]:
        self._update_postprocessor(inc_dataset)

    if self._cfg["infer_head"] == 'NCM':
        self._ex.logger.info("compute prototype")
        self.update_prototype()

    if self._memory_size.memsize != 0:
        self._ex.logger.info("build memory")
        self.build_exemplars(inc_dataset, self._coreset_strategy)

        if self._cfg["save_mem"]:
            save_path = os.path.join(os.getcwd(), "ckpts/mem")
            memory = {
                'x': inc_dataset.data_memory,
                'y': inc_dataset.targets_memory,
                'herding': self._herding_matrix
            }
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not (os.path.exists(f"{save_path}/mem_step{self._task}.ckpt") and self._cfg['load_mem']):
                torch.save(memory, "{}/mem_step{}.ckpt".format(save_path, self._task))
                self._ex.logger.info(f"Save step{self._task} memory!")

    self._parallel_network.eval()
    self._old_model = deepcopy(self._parallel_network)
    self._old_model.module.freeze()
    del self._inc_dataset.shared_data_inc
    self._inc_dataset.shared_data_inc = None