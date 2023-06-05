# from argparse import Namespace
# import copy
# import logging
# import os
# import wandb


# class Logger:

#     def __init__(self, args: Namespace):
#         self.args = args

#         if args.log_tool == "wandb":
#             self._init_wandb()
#             self.logger = wandb

#         elif args.log_tool == "logging":
#             logging.basicConfig(filename=args.proj_path / 'log' / (args.exp_name+'.log'),
#                                 filemode='w',
#                                 level=logging.INFO,
#                                 force=True)
#             self.logger = logging.getLogger()

#         elif args.log_tool == "all":

#             logging.basicConfig(filename=args.proj_path / 'log' / (args.exp_name+'.log'),
#                                 filemode='w',
#                                 level=logging.INFO,
#                                 force=True)
#             self.logger = logging.getLogger()

#         else:
#             raise ValueError("log_tool must be one of wandb, logging, all")


#     def _init_wandb(self):
#         wandb.init(
#                 project="leit",
#                 config=copy.deepcopy(dict(self.args._get_kwargs())),
#                 group="_".join(tags),
#                 tags=tags,
#                 name="r"+str(self.args.random_state),
#                 settings=wandb.Settings(_service_wait=60))
#         # initialize weight and bias
#         if self.args.model_type != "initialize":
#             os.environ["WANDB_MODE"] = "dryrun"
#         tags = [
#             self.args.ml_task,
#             self.args.data,
#             self.args.leit_model,
#         ]
#         if self.args.leit_model == "ivp_vae" or self.args.leit_model == "red_vae":
#             tags.append(self.args.ivp_solver)
#         tags.append(self.args.test_info)
