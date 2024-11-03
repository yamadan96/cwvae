import numpy as np
import argparse
import yaml
import os

import tensorflow.compat.v1 as tf

from cwvae import build_model
from loggers.summary import Summary
from loggers.checkpoint import Checkpoint
from data_loader import *
import tools

import wandb  # 追加

def train_setup(cfg, loss):
    session_config = tf.ConfigProto(device_count={"GPU": 1}, log_device_placement=False)
    session = tf.Session(config=session_config)
    step = tools.Step(session)

    with tf.name_scope("optimizer"):
        # Getting all trainable variables.
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        print(f"Number of trainable variables: {len(weights)}")  # デバッグ出力

        # Creating optimizer.
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr, epsilon=1e-04)
        print("Optimizer initialized.")  # デバッグ出力

        # Computing gradients.
        grads = optimizer.get_gradients(loss, weights)
        grad_norm = tf.global_norm(grads)
        print("Gradients computed.")  # デバッグ出力

        # Clipping gradients by global norm, and applying gradient.
        if cfg.clip_grad_norm_by is not None:
            capped_grads, grad_norm = tf.clip_by_global_norm(grads, cfg.clip_grad_norm_by)
            capped_gvs = [
                (capped_grads[i], weights[i]) for i in range(len(weights))
            ]
            apply_grads = optimizer.apply_gradients(capped_gvs)
            print("Gradients clipped and optimizer applied with capped gradients.")  # デバッグ出力
        else:
            gvs = zip(grads, weights)
            apply_grads = optimizer.apply_gradients(gvs)
            print("Optimizer applied without gradient clipping.")  # デバッグ出力
    return apply_grads, grad_norm, session, step

if __name__ == "__main__":
    tf.disable_v2_behavior()
    print("TensorFlow v2 behavior disabled.")  # デバッグ出力

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default=None,
        type=str,
        help="path to root log directory",
    )
    parser.add_argument(
        "--datadir",
        default=None,
        type=str,
        help="path to root data directory",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="path to config yaml file",
        required=True,
    )
    parser.add_argument(
        "--base-config",
        default="./configs/base_config.yml",
        type=str,
        help="path to base config yaml file",
    )

    args = parser.parse_args()
    print(f"Arguments received: {args}")  # デバッグ出力

    cfg = tools.read_configs(
        args.config, args.base_config, datadir=args.datadir, logdir=args.logdir
    )
    print(f"Configuration loaded: {cfg}")  # デバッグ出力

    # wandb の初期化
    wandb.init(project="cwvae-2")  # プロジェクト名を適宜変更
    wandb.config.update(dict(cfg))  # ハイパーパラメータをログ
    print("wandb initialized.")  # デバッグ出力

    # Creating model dir with experiment name.
    exp_rootdir = os.path.join(cfg.logdir, cfg.dataset, tools.exp_name(cfg))
    os.makedirs(exp_rootdir, exist_ok=True)
    print(f"Experiment directory created at: {exp_rootdir}")  # デバッグ出力

    # Dumping config.
    print(f"Configuration: {cfg}")
    with open(os.path.join(exp_rootdir, "config.yml"), "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)
    print("Configuration saved to config.yml")  # デバッグ出力

    # Load dataset.
    try:
        train_data_batch, val_data_batch = load_dataset(cfg)
        print("Dataset loaded successfully.")  # デバッグ出力
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    # Build model.
    try:
        model_components = build_model(cfg)
        model = model_components["meta"]["model"]
        print("Model built successfully.")  # デバッグ出力
    except Exception as e:
        print(f"Error building model: {e}")
        exit(1)

    # Setting up training.
    try:
        apply_grads, grad_norm, session, step = train_setup(cfg, model.loss)
        print("Training setup complete.")  # デバッグ出力
    except Exception as e:
        print(f"Error setting up training: {e}")
        exit(1)

    # Define summaries.
    summary = Summary(exp_rootdir, save_gifs=cfg.save_gifs)
    summary.build_summary(cfg, model_components, grad_norm=grad_norm)
    print("Summaries defined.")  # デバッグ出力

    # Define checkpoint saver for variables currently in session.
    checkpoint = Checkpoint(exp_rootdir)
    print("Checkpoint saver initialized.")  # デバッグ出力

    # Restore model (if exists).
    if os.path.exists(checkpoint.log_dir_model):
        print(f"Restoring model from {checkpoint.log_dir_model}")
        checkpoint.restore(session)
        print(f"Will start training from step {step()}")  # デバッグ出力
    else:
        # Initialize all variables.
        session.run(tf.global_variables_initializer())
        print("Variables initialized.")  # デバッグ出力

    # Start training.
    print("Getting validation batches.")
    try:
        val_batches = get_multiple_batches(val_data_batch, cfg.num_val_batches, session)
        print("Validation batches obtained.")  # デバッグ出力
    except Exception as e:
        print(f"Error getting validation batches: {e}")
        exit(1)

    print("Training started.")
    while True:
        try:
            train_batch = get_single_batch(train_data_batch, session)
            feed_dict_train = {model_components["training"]["obs"]: train_batch}
            feed_dict_val = {model_components["training"]["obs"]: val_batches}

            # Train one step.
            session.run(fetches=apply_grads, feed_dict=feed_dict_train)

            # 損失値の計算
            loss_value = session.run(model.loss, feed_dict=feed_dict_train)
            print(f"Step {step()}: Loss = {loss_value}")  # デバッグ出力

            # wandb にログ
            wandb.log({"loss": loss_value}, step=step())
            print(f"wandb logged at step {step()}")  # デバッグ出力

            # Saving scalar summaries.
            if step() % cfg.save_scalars_every == 0:
                print(f"Saving scalar summaries at step {step()}")  # デバッグ出力
                summaries = session.run(
                    summary.scalar_summary, feed_dict=feed_dict_train
                )
                summary.save(summaries, step(), True)
                summaries = session.run(summary.scalar_summary, feed_dict=feed_dict_val)
                summary.save(summaries, step(), False)

            # Saving gif summaries.
            if step() % cfg.save_gifs_every == 0:
                print(f"Saving GIF summaries at step {step()}")  # デバッグ出力
                summaries = session.run(summary.gif_summary, feed_dict=feed_dict_train)
                summary.save(summaries, step(), True)
                summaries = session.run(summary.gif_summary, feed_dict=feed_dict_val)
                summary.save(summaries, step(), False)

            # Saving model.
            if step() % cfg.save_model_every == 0:
                print(f"Saving model checkpoint at step {step()}")  # デバッグ出力
                checkpoint.save(session)

            if cfg.save_named_model_every and step() % cfg.save_named_model_every == 0:
                print(f"Saving named model checkpoint at step {step()}")  # デバッグ出力
                checkpoint.save(session, save_dir="model_{}".format(step()))

            step.increment()
        except tf.errors.OutOfRangeError:
            print("End of dataset reached.")  # デバッグ出力
            break
        except Exception as e:
            print(f"Error during training at step {step()}: {e}")
            exit(1)

    print("Training complete.")
    wandb.finish()  # wandb セッションの終了
