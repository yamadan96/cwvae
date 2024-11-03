import numpy as np
import argparse
import yaml
import os

import tensorflow.compat.v1 as tf
from tqdm import tqdm  # tqdm をインポート

from cwvae import build_model
from loggers.summary import Summary
from loggers.checkpoint import Checkpoint
from data_loader import load_dataset, get_multiple_batches, get_single_batch
import tools

import wandb  # wandb をインポート

def train_setup(cfg, loss):
    session_config = tf.ConfigProto(device_count={"GPU": 1}, log_device_placement=False)
    session = tf.Session(config=session_config)
    step = tools.Step(session)

    with tf.name_scope("optimizer"):
        # トレーニング可能な変数を取得
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        print(f"Number of trainable variables: {len(weights)}")  # デバッグ出力

        # オプティマイザを作成
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr, epsilon=1e-04)
        print("Optimizer initialized.")  # デバッグ出力

        # 勾配を計算
        grads = optimizer.get_gradients(loss, weights)
        grad_norm = tf.global_norm(grads)
        print("Gradients computed.")  # デバッグ出力

        # 勾配のクリッピングと適用
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
    wandb.init(project="cwvae-2", config=dict(cfg))  # プロジェクト名を適宜変更
    print("wandb initialized.")  # デバッグ出力

    # 実験ディレクトリの作成
    exp_rootdir = os.path.join(cfg.logdir, cfg.dataset, tools.exp_name(cfg))
    os.makedirs(exp_rootdir, exist_ok=True)
    print(f"Experiment directory created at: {exp_rootdir}")  # デバッグ出力

    # 設定の保存
    print(f"Configuration: {cfg}")
    with open(os.path.join(exp_rootdir, "config.yml"), "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)
    print("Configuration saved to config.yml")  # デバッグ出力

    # データセットのロード
    try:
        train_data_batch, val_data_batch = load_dataset(cfg)
        print("Dataset loaded successfully.")  # デバッグ出力
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    # モデルの構築
    try:
        model_components = build_model(cfg)
        model = model_components["meta"]["model"]
        print("Model built successfully.")  # デバッグ出力
    except Exception as e:
        print(f"Error building model: {e}")
        exit(1)

    # トレーニングのセットアップ
    try:
        apply_grads, grad_norm, session, step = train_setup(cfg, model.loss)
        print("Training setup complete.")  # デバッグ出力
    except Exception as e:
        print(f"Error setting up training: {e}")
        exit(1)

    # サマリーの定義
    summary = Summary(exp_rootdir, save_gifs=cfg.save_gifs)
    summary.build_summary(cfg, model_components, grad_norm=grad_norm)
    print("Summaries defined.")  # デバッグ出力

    # チェックポイントセーバーの定義
    checkpoint = Checkpoint(exp_rootdir)
    print("Checkpoint saver initialized.")  # デバッグ出力

    # モデルのリストア
    if os.path.exists(checkpoint.log_dir_model):
        print(f"Restoring model from {checkpoint.log_dir_model}")
        checkpoint.restore(session)
        print(f"Will start training from step {step()}")  # デバッグ出力
    else:
        # 変数の初期化
        session.run(tf.global_variables_initializer())
        print("Variables initialized.")  # デバッグ出力

    # 検証データのイテレータを初期化
    val_data_batch.initialize(session)
    print("Getting validation batches.")
    try:
        val_batches = get_multiple_batches(val_data_batch.get_batch(), cfg.num_val_batches, session)
        print("Validation batches obtained.")  # デバッグ出力
    except Exception as e:
        print(f"Error getting validation batches: {e}")
        exit(1)

    print("Training started.")

    # tqdm を使用して進行状況を表示
    total_steps = cfg.total_steps if hasattr(cfg, 'total_steps') else 100000
    with tqdm(total=total_steps, desc="Training") as pbar:
        while True:
            try:
                # トレーニングデータのイテレータを初期化（必要に応じて）
                train_data_batch.initialize(session)

                train_batch = get_single_batch(train_data_batch.get_batch(), session)
                feed_dict_train = {model_components["training"]["obs"]: train_batch}
                feed_dict_val = {model_components["training"]["obs"]: val_batches}

                # トレーニングステップと損失値の取得
                _, loss_value, nll_value, kl_value, grad_norm_value = session.run(
                    [apply_grads, model.loss, model._nll_term, model._kl_term, grad_norm],
                    feed_dict=feed_dict_train
                )

                # wandb にログ
                wandb.log({
                    "loss": loss_value,
                    "nll_term": nll_value,
                    "kl_term": kl_value,
                    "grad_norm": grad_norm_value,
                    "step": step()
                })

                # プログレスバーの更新
                pbar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "nll": f"{nll_value:.4f}",
                    "kl": f"{kl_value:.4f}"
                })
                pbar.update(1)

                # スカラーサマリーの保存
                if step() % cfg.save_scalars_every == 0:
                    print(f"Saving scalar summaries at step {step()}")  # デバッグ出力
                    summaries = session.run(
                        summary.scalar_summary, feed_dict=feed_dict_train
                    )
                    summary.save(summaries, step(), True)
                    summaries = session.run(summary.scalar_summary, feed_dict=feed_dict_val)
                    summary.save(summaries, step(), False)

                # GIF サマリーの保存
                if step() % cfg.save_gifs_every == 0:
                    print(f"Saving GIF summaries at step {step()}")  # デバッグ出力
                    summaries = session.run(summary.gif_summary, feed_dict=feed_dict_train)
                    summary.save(summaries, step(), True)
                    summaries = session.run(summary.gif_summary, feed_dict=feed_dict_val)
                    summary.save(summaries, step(), False)

                # モデルの保存
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
                break  # エラー発生時にループを抜ける

    print("Training complete.")
    wandb.finish()  # wandb セッションの終了
