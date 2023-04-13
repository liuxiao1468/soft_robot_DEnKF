import torch
import numpy as np

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from soft_robot.config import cfg
from soft_robot.dataset.sw_dataset import SmartwatchDataset
from soft_robot.model.DEnKF import EnsembleKfNoAction
from soft_robot.optimizer.lr_scheduler import build_lr_scheduler
from soft_robot.optimizer.optimizer import build_optimizer


class SwEngine:
    """
    SmartwatchEngine.
    This class is in parts copied from the original engine.py.
    It has been adapted to handle the smartwatch data set.
    Some parameters of the input "args" config are handled differently:
       * log_freq, eval_freq, and save_freq all refer the global step count
       * the test function reports the same loss as the train function, but over the test dataset
       * this Engine does not allow masking and the consideration of actions
    """

    def __init__(self, args, logger):
        """
        Constructor
        Args:
            args: the config params read in train.py
            logger: a logger instance to log updates and warnings
        """

        self.__args = args
        self.__logger = logger

        # global step counter across all epochs
        self.__global_step = 0

        # create model with params from config
        self.__model = EnsembleKfNoAction(
            num_ensemble=args.train.num_ensemble,
            dim_x=args.train.dim_x,
            dim_z=args.train.dim_z,
            input_size=args.train.input_size
        )

        # move to GPU if possible
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.__model.cuda()

        # tensorboard writer
        tb_path = Path(args.train.log_directory) / args.train.model_name / "tensorboard"
        tb_path.mkdir(parents=True, exist_ok=True)
        self.__tb_writer = SummaryWriter(tb_path)

        # prepare data sets such that they only have to load once
        self.__train_dataset = SmartwatchDataset(
            dataset_path=args.train.data_path,
            num_ensemble=args.train.num_ensemble,
            dim_x=args.train.dim_x,
            dim_z=args.train.dim_z
        )
        self.__test_dataset = SmartwatchDataset(
            dataset_path=args.test.data_path,
            num_ensemble=args.test.num_ensemble,
            dim_x=args.test.dim_x,
            dim_z=args.test.dim_z
        )
        self.__eval_dataset = SmartwatchDataset(
            dataset_path=args.test.eval_data_path,
            num_ensemble=args.test.num_ensemble,
            dim_x=args.train.dim_x,
            dim_z=args.train.dim_z
        )

    def eval_model(self, checkpoint_path=None):
        """
        evaluate the current model of this class on the dest data using the parameters from self.__args
        """

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.__model.load_state_dict(checkpoint["model"])
            self.__logger.info("loaded model from {}".format(checkpoint_path))

        # the eval test iterates one data set in sequential order
        eval_dataloader = DataLoader(self.__eval_dataset, batch_size=1, shuffle=False)

        # set model to evaluation mode before the predictions
        self.__model.eval()

        losses = []
        ensemble_pred = None
        step = 0
        preds = []
        gts = []
        for state_gt, state_pre_ens, obs in eval_dataloader:
            state_gt = state_gt.to(self.__device)
            state_pre_ens = state_pre_ens.to(self.__device)
            obs = obs.to(self.__device)

            # only get the initial previous state, then use the predictions
            if ensemble_pred is None:
                ensemble_pred = state_pre_ens

            with torch.no_grad():
                output = self.__model(state_old_ens=ensemble_pred, raw_obs=obs)
                state_pred = output[1]  # -> state estimation
                ensemble_pred = output[0]  # -> ensemble estimation

            # calculate loss
            gt_xyz = state_gt[0, 0, :3]
            gts.append(gt_xyz.cpu().numpy())
            pr_xyz = state_pred[0, 0, :3]
            preds.append(pr_xyz.cpu().numpy())

            # euclidian distance
            dist = torch.sqrt(torch.sum(torch.square(pr_xyz - gt_xyz)))
            losses.append(dist.cpu().item())

            # update step and verbose
            step += 1
            if step % 1000 == 0:
                self.__logger.info("[eval]: [{}], loss: {:.12f}".format(step, np.mean(losses)))

        return np.array(gts), np.array(preds)

    def test_model(self):
        """
        evaluate the current model of this class on the dest data using the parameters from self.__args
        """

        batch_size = self.__args.train.batch_size
        test_dataloader = DataLoader(self.__test_dataset, batch_size=batch_size, shuffle=True)

        # The loss function
        mse_criterion = torch.nn.MSELoss()

        # set model to evaluation mode before the predictions
        self.__model.eval()

        losses = []
        for state_gt, state_pre_ens, obs in test_dataloader:
            state_gt = state_gt.to(self.__device)
            state_pre_ens = state_pre_ens.to(self.__device)
            obs = obs.to(self.__device)

            with torch.no_grad():
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
                losses.append(final_loss.cpu().item())

        self.__logger.info("[test result][gs]: [{}], loss: {:.12f}".format(
            self.__global_step,
            np.mean(losses)
        ))

    def train_model(self):
        """
        train model on entire training data set using the parameters from self.__args
        """
        # create data loader from data set
        batch_size = self.__args.train.batch_size
        train_data_loader = DataLoader(self.__train_dataset, batch_size=batch_size, shuffle=True)

        # The loss function
        mse_criterion = torch.nn.MSELoss()

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

        # the training loop
        for epoch in range(self.__args.train.num_epochs):
            # reset local step counter for every epoch
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
                if epoch <= 15:
                    # train the sensor model first
                    final_loss = loss_3
                else:
                    # end-to-end mode, now all models ar optimized
                    final_loss = loss_1 + loss_2 + loss_3 + loss_4

                if np.isnan(final_loss.cpu().item()):
                    raise UserWarning("NaN in loss occurred. Aborting training.")

                # back prop
                final_loss.backward()
                optimizer.step()
                current_lr = optimizer.param_groups[0]["lr"]

                # update steps and LR-scheduler
                step += 1
                self.__global_step += 1
                scheduler.step(self.__global_step)

                # verbose and tensorboard
                if self.__global_step % self.__args.train.log_freq == 0:
                    self.__logger.info(
                        "[epoch][s/gs]: [{}][{}/{}], lr: {:.12f}, loss: {:.12f}".format(
                            epoch,
                            step,
                            self.__global_step,
                            current_lr,
                            final_loss
                        )
                    )
                    self.__tb_writer.add_scalar("end_to_end_loss", final_loss.cpu().item(), self.__global_step)
                    self.__tb_writer.add_scalar("transition model", loss_2.cpu().item(), self.__global_step)
                    self.__tb_writer.add_scalar("sensor_model", loss_3.cpu().item(), self.__global_step)
                    self.__tb_writer.add_scalar("observation_model", loss_4.cpu().item(), self.__global_step)
                    # self.writer.add_scalar('learning_rate', current_lr, self.global_step)

                # Save a model based of a chosen save frequency
                if self.__global_step % self.__args.train.save_freq == 0:
                    save_path = Path(
                        self.__args.train.log_directory) / self.__args.train.model_name / "checkpoint-{}".format(
                        self.__global_step)
                    checkpoint = {
                        "global_step": self.__global_step,
                        "model": self.__model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(checkpoint, save_path)
                    self.__logger.info(
                        "[save model][s/gs]: [{}][{}/{}], saved checkpoint to {}".format(
                            epoch,
                            step,
                            self.__global_step,
                            save_path
                        )
                    )

                # get loss on test data if online evaluation is active
                if self.__args.mode.do_online_eval:
                    if self.__global_step % self.__args.train.eval_freq == 0:
                        self.test_model()
