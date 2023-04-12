import os
import time
import pickle
import torch

import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from soft_robot.dataset.sw_dataset import SmartwatchDataset
from soft_robot.model.DEnKF import EnsembleKfNoAction
from soft_robot.optimizer.lr_scheduler import build_lr_scheduler
from soft_robot.optimizer.optimizer import build_optimizer


class SwEngine:
    """
    SmartwatchEngine.
    This class is in parts copied from the original engine.py.
    It has been adapted to handle the smartwatch data set
    """

    def __init__(self, args, logger):

        self.__args = args
        self.__logger = logger

        self.__global_step = 0

        # create model with params from config
        self.__model = EnsembleKfNoAction(
            num_ensemble=args.train.num_ensemble,
            dim_x=args.train.dim_x,
            dim_z=args.train.dim_z,
            input_size=args.train.input_size
        )

        # Check model type
        if not isinstance(self.__model, torch.nn.Module):
            raise TypeError("model must be an instance of nn.Module")

        # move to GPU if possible
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.__model.cuda()

        # tensorboard writer
        self.__writer = SummaryWriter(
            f"./experiments/{self.__args.train.model_name}/summaries"
        )

        # prepare data sets such that they only have to load once
        self.__test_dataset = SmartwatchDataset(
            dataset_path=self.__args.test.data_path,
            num_ensemble=self.__args.test.num_ensemble,
            dim_x=self.__args.test.dim_x,
            dim_z=self.__args.test.dim_z
        )
        self.__train_dataset = SmartwatchDataset(
            dataset_path=self.__args.train.data_path,
            num_ensemble=self.__args.train.num_ensemble,
            dim_x=self.__args.train.dim_x,
            dim_z=self.__args.train.dim_z
        )

    def test(self):

        test_dataloader = DataLoader(self.__test_dataset, batch_size=1, shuffle=False)

        step = 0
        data = {}
        data_save = []
        ensemble_save = []
        gt_save = []
        obs_save = []

        # set model to evaluation mode before the predictions
        self.__model.eval()

        last_ensemble, state = None, None
        for state_gt, state_pre_ens, obs in test_dataloader:
            state_gt = state_gt.to(self.__device)
            state_pre_ens = state_pre_ens.to(self.__device)
            obs = obs.to(self.__device)

            if last_ensemble is None:
                last_ensemble = state_pre_ens

            with torch.no_grad():
                output = self.__model(state_old=last_ensemble, raw_obs=obs)
                last_ensemble = output[0]  # -> ensemble estimation
                state = output[1]  # -> final estimation
                obs_p = output[3]  # -> learned observation

                final_ensemble = last_ensemble  # -> make sure these variables are tensor
                final_est = state
                obs_est = obs_p

                final_ensemble = final_ensemble.cpu().detach().numpy()
                final_est = final_est.cpu().detach().numpy()
                obs_est = obs_est.cpu().detach().numpy()
                state_gt = state_gt.cpu().detach().numpy()

                data_save.append(final_est)
                ensemble_save.append(final_ensemble)
                gt_save.append(state_gt)
                obs_save.append(obs_est)
                step = step + 1

        data["state"] = data_save
        data["ensemble"] = ensemble_save
        data["gt"] = gt_save
        data["observation"] = obs_save

        save_path = os.path.join(
            self.__args.train.eval_summary_directory,
            self.__args.train.model_name,
            "eval-result-{}.pkl".format(self.__global_step),
        )

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    # @TODO: needs more work but is not in use yet
    def online_test(self):

        self.__model.eval()

        test_dataloader = torch.utils.data.DataLoader(self.__test_dataset, batch_size=1, shuffle=False)

        step = 0
        data = {}
        data_save = []
        ensemble_save = []
        gt_save = []
        obs_save = []
        ensemble = None
        for state_gt, state_pre_ens, obs in test_dataloader:
            state_gt = state_gt.to(self.__device)
            state_pre_ens = state_pre_ens.to(self.__device)
            obs = obs.to(self.__device)

            with torch.no_grad():
                if ensemble is None:
                    ensemble = state_pre_ens
                output = self.__model(state_old=ensemble, raw_obs=obs)
                ensemble = output[0]  # -> ensemble estimation
                state = output[1]  # -> final estimation
                obs_p = output[3]  # -> learned observation
                if step % 1000 == 0:
                    print("===============")
                    print(state)
                    print(obs_p)
                    print(state_gt)
                    print(output[2])

                final_ensemble = ensemble  # -> make sure these variables are tensor
                final_est = state
                obs_est = obs_p

                final_ensemble = final_ensemble.cpu().detach().numpy()
                final_est = final_est.cpu().detach().numpy()
                obs_est = obs_est.cpu().detach().numpy()
                state_gt = state_gt.cpu().detach().numpy()

                data_save.append(final_est)
                ensemble_save.append(final_ensemble)
                gt_save.append(state_gt)
                obs_save.append(obs_est)
                step = step + 1

        data["state"] = data_save
        data["ensemble"] = ensemble_save
        data["gt"] = gt_save
        data["observation"] = obs_save

        save_path = os.path.join(
            self.__args.train.eval_summary_directory,
            self.__args.train.model_name,
            "test-result-{}.pkl".format("52"),
        )

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def train(self):

        batch_size = self.__args.train.batch_size
        train_data_loader = DataLoader(self.__train_dataset, batch_size=batch_size, shuffle=True)

        pytorch_total_params = sum(p.numel() for p in self.__model.parameters() if p.requires_grad)
        self.__logger.info("Total number of parameters: {}".format(pytorch_total_params))

        # Create optimizer
        optimizer = build_optimizer(
            [
                self.__model.process_model,
                self.__model.observation_model,
                self.__model.observation_noise,
                self.__model.sensor_model,
            ],
            self.__args.network.name,
            self.__args.optim.optim,
            self.__args.train.learning_rate,
            self.__args.train.weight_decay,
            self.__args.train.adam_eps,
        )

        # Create LR scheduler
        num_total_steps = self.__args.train.num_epochs * len(train_data_loader)
        scheduler = build_lr_scheduler(
            optimizer,
            self.__args.optim.lr_scheduler,
            self.__args.train.learning_rate,
            num_total_steps,
            self.__args.train.end_learning_rate,
        )

        # The loss function
        mse_criterion = torch.nn.MSELoss()

        # Epoch calculations
        steps_per_epoch = len(train_data_loader)
        epoch = self.__global_step // steps_per_epoch
        duration = 0

        ####################################################################################################
        # MAIN TRAINING LOOP
        ####################################################################################################

        while epoch < self.__args.train.num_epochs:
            step = 0

            # ensure model is in training mode
            self.__model.train()

            for state_gt, state_pre_ens, obs in train_data_loader:

                # move all to GPU
                state_gt = state_gt.to(self.__device)
                state_pre_ens = state_pre_ens.to(self.__device)
                obs = obs.to(self.__device)

                # define the training curriculum
                optimizer.zero_grad()
                before_op_time = time.time()

                # forward pass
                output = self.__model(state_old_ens=state_pre_ens, raw_obs=obs)
                final_est = output[1]  # -> final estimation
                inter_est = output[2]  # -> state transition output
                obs_est = output[3]  # -> learned observation
                hx = output[5]  # -> observation output

                # calculate loss
                loss_1 = mse_criterion(final_est, state_gt)
                loss_2 = mse_criterion(inter_est, state_gt)
                loss_3 = mse_criterion(obs_est, state_gt)
                loss_4 = mse_criterion(hx, state_gt)

                final_loss = loss_1 + loss_2 + loss_3 + loss_4

                # back prop
                final_loss.backward()
                optimizer.step()
                current_lr = optimizer.param_groups[0]["lr"]

                # verbose
                if self.__global_step % self.__args.train.log_freq == 0:
                    string = "[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}"
                    self.__logger.info(
                        string.format(
                            epoch,
                            step,
                            steps_per_epoch,
                            self.__global_step,
                            current_lr,
                            final_loss,
                        )
                    )
                    if np.isnan(final_loss.cpu().item()):
                        self.__logger.warning("NaN in loss occurred. Aborting training.")
                        return -1

                # tensorboard
                duration += time.time() - before_op_time
                if (
                        self.__global_step
                        and self.__global_step % self.__args.train.log_freq == 0
                ):
                    self.__writer.add_scalar(
                        "end_to_end_loss", final_loss.cpu().item(), self.__global_step
                    )
                    self.__writer.add_scalar(
                        "transition model", loss_2.cpu().item(), self.__global_step
                    )
                    self.__writer.add_scalar(
                        "sensor_model", loss_3.cpu().item(), self.__global_step
                    )
                    self.__writer.add_scalar(
                        "observation_model", loss_4.cpu().item(), self.__global_step
                    )
                    # self.writer.add_scalar('learning_rate', current_lr, self.global_step)

                step += 1
                self.__global_step += 1
                if scheduler is not None:
                    scheduler.step(self.__global_step)

            # Save a model based of a chosen save frequency
            if self.__global_step != 0 and (epoch + 1) % self.__args.train.save_freq == 0:
                checkpoint = {
                    "global_step": self.__global_step,
                    "model": self.__model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(
                        self.__args.train.log_directory,
                        self.__args.train.model_name,
                        "final-model-{}".format(self.__global_step),
                    ),
                )

            # online evaluation
            if (
                    self.__args.mode.do_online_eval
                    and self.__global_step != 0
                    and epoch + 1 >= 50
                    and (epoch + 1) % self.__args.train.eval_freq == 0
            ):
                time.sleep(0.1)
                self.test()

            # Update epoch
            epoch += 1
