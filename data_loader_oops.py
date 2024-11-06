import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import os
import cv2

class OopsDataset:
    def __init__(self, batch_size, epochs, train=True, seq_len=100, data_root=None):
        self._train = train
        self._batch_size = batch_size
        self._epochs = epochs
        self._seq_len = seq_len  # 必要なシーケンス長は100フレーム

        if self._train:
            data_dir = os.path.join(data_root, 'train')
        else:
            data_dir = os.path.join(data_root, 'test')

        # データセットの準備
        ds = self._create_dataset(data_dir)
        ds = ds.shuffle(1000)  # シーケンス全体でシャッフル
        ds = ds.repeat(self._epochs)
        ds = ds.batch(self._batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        self.batch = tf.data.make_one_shot_iterator(ds).get_next()

    def get_batch(self):
        return self.batch

    def _create_dataset(self, data_dir):
        # mp4ファイルのリストを取得し、バイト文字列にエンコード
        video_files = [os.path.join(data_dir, f).encode('utf-8') for f in os.listdir(data_dir) if f.endswith('.mp4')]

        # ジェネレータ関数の定義
        def generator():
            import numpy as np
            import cv2

            for video_file in video_files:
                video_path = video_file.decode('utf-8')
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"ビデオファイルを開くことができませんでした: {video_path}")
                    continue
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (64, 64))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                cap.release()

                num_frames = len(frames)
                frames = np.array(frames, dtype=np.float32) / 255.0  # ピクセル値を正規化

                # ビデオがseq_lenに満たない場合もゼロパディングして含める
                if num_frames < self._seq_len:
                    padding = np.zeros((self._seq_len - num_frames, 64, 64, 3), dtype=np.float32)
                    frames = np.concatenate([frames, padding], axis=0)
                    yield frames
                else:
                    # スライディングウィンドウでシーケンスを生成
                    stride = self._seq_len // 2
                    starts = range(0, num_frames - self._seq_len + 1, stride)
                    for start in starts:
                        end = start + self._seq_len
                        seq = frames[start:end]

                        # シーケンスがseq_lenに満たない場合はゼロパディング
                        if len(seq) < self._seq_len:
                            padding = np.zeros((self._seq_len - len(seq), 64, 64, 3), dtype=np.float32)
                            seq = np.concatenate([seq, padding], axis=0)
                        
                        yield seq

        output_types = tf.float32
        output_shapes = (self._seq_len, 64, 64, 3)
        ds = tf.data.Dataset.from_generator(generator, output_types=output_types, output_shapes=output_shapes)
        return ds



class MineRL:
    def __init__(self, batch_size, epochs, train=True, seq_len=None, data_root=None):
        self._train = train
        self._batch_size = batch_size
        self._epochs = epochs
        self._data_seq_len = 500
        self._seq_len = seq_len
        if self._train:
            ds = tfds.load("minerl_navigate", data_dir=data_root, shuffle_files=True)[
                "train"
            ]
        else:
            ds = tfds.load("minerl_navigate", data_dir=data_root, shuffle_files=False)[
                "test"
            ]
        ds = ds.map(lambda vid: vid["video"]).flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices(self._process_seq(x))
        )
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.repeat(self._epochs)
        if self._train:
            ds = ds.shuffle(10 * self._batch_size)
        ds = ds.batch(self._batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        self.batch = tf.data.make_one_shot_iterator(ds).get_next()

    def get_batch(self):
        return self.batch

    def _process_seq(self, seq):
        if self._seq_len:
            seq_len_tr = self._data_seq_len - (self._data_seq_len % self._seq_len)
            seq = seq[:seq_len_tr]
            seq = tf.reshape(
                seq,
                tf.concat(
                    [[seq_len_tr // self._seq_len, self._seq_len], tf.shape(seq)[1:]],
                    -1,
                ),
            )
        else:
            seq = tf.expand_dims(seq, 0)
        seq = tf.cast(seq, tf.float32) / 255.0
        return seq


class GQNMazes:
    def __init__(self, batch_size, epochs, train=True, seq_len=None, data_root=None):
        self._train = train
        self._batch_size = batch_size
        self._epochs = epochs
        self._data_seq_len = 300
        self._seq_len = seq_len
        if self._train:
            ds = tfds.load("gqn_mazes", data_dir=data_root, shuffle_files=True)["train"]
        else:
            ds = tfds.load("gqn_mazes", data_dir=data_root, shuffle_files=False)["test"]
        ds = ds.map(lambda vid: vid["video"]).flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices(self._process_seq(x))
        )
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.repeat(self._epochs)
        if self._train:
            ds = ds.shuffle(10 * self._batch_size)
        ds = ds.batch(self._batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        self.batch = tf.data.make_one_shot_iterator(ds).get_next()

    def get_batch(self):
        return self.batch

    def _process_seq(self, seq):
        if self._seq_len:
            seq_len_tr = self._data_seq_len - (self._data_seq_len % self._seq_len)
            seq = seq[:seq_len_tr]
            seq = tf.reshape(
                seq,
                tf.concat(
                    [[seq_len_tr // self._seq_len, self._seq_len], tf.shape(seq)[1:]],
                    -1,
                ),
            )
        else:
            seq = tf.expand_dims(seq, 0)
        seq = tf.cast(seq, tf.float32) / 255.0
        return seq


class MovingMNIST:
    def __init__(self, batch_size, epochs, train=True, seq_len=None, data_root=None):
        self._train = train
        self._batch_size = batch_size
        self._epochs = epochs
        if self._train:
            self._data_seq_len = 100
        else:
            self._data_seq_len = 1000
        self._seq_len = seq_len
        if self._train:
            ds = tfds.load(
                "moving_mnist_2digit", data_dir=data_root, shuffle_files=True
            )["train"]
        else:
            ds = tfds.load(
                "moving_mnist_2digit", data_dir=data_root, shuffle_files=False
            )["test"]
        ds = ds.map(lambda vid: vid["video"]).flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices(self._process_seq(x))
        )
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.repeat(self._epochs)
        if self._train:
            ds = ds.shuffle(10 * self._batch_size)
        ds = ds.batch(self._batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        self.batch = tf.data.make_one_shot_iterator(ds).get_next()

    def get_batch(self):
        return self.batch

    def _process_seq(self, seq):
        if self._seq_len:
            seq_len_tr = self._data_seq_len - (self._data_seq_len % self._seq_len)
            seq = seq[:seq_len_tr]
            seq = tf.reshape(
                seq,
                tf.concat(
                    [[seq_len_tr // self._seq_len, self._seq_len], tf.shape(seq)[1:]],
                    -1,
                ),
            )
        else:
            seq = tf.expand_dims(seq, 0)
        seq = tf.cast(seq, tf.float32) / 255.0
        return seq


def load_dataset(cfg, **kwargs):
    if cfg.dataset == "oops":
        train_data_batch = OopsDataset(
            cfg.batch_size,
            cfg.num_epochs,
            train=True,
            seq_len=cfg.seq_len,
            data_root=cfg.datadir,
        ).get_batch()
        test_data_batch = OopsDataset(
            cfg.batch_size,
            1,
            train=False,
            seq_len=cfg.eval_seq_len,
            data_root=cfg.datadir,
        ).get_batch()   
    elif cfg.dataset == "minerl":
        import minerl_navigate

        train_data_batch = MineRL(
            cfg.batch_size,
            cfg.num_epochs,
            train=True,
            seq_len=cfg.seq_len,
            data_root=cfg.datadir,
        ).get_batch()
        test_data_batch = MineRL(
            cfg.batch_size,
            1,
            train=False,
            seq_len=cfg.eval_seq_len,
            data_root=cfg.datadir,
        ).get_batch()
    elif cfg.dataset == "mmnist":
        import datasets.moving_mnist

        train_data_batch = MovingMNIST(
            cfg.batch_size,
            cfg.num_epochs,
            train=True,
            seq_len=cfg.seq_len,
            data_root=cfg.datadir,
        ).get_batch()
        test_data_batch = MovingMNIST(
            cfg.batch_size,
            1,
            train=False,
            seq_len=cfg.eval_seq_len,
            data_root=cfg.datadir,
        ).get_batch()
    elif cfg.dataset == "mazes":
        import datasets.gqn_mazes

        train_data_batch = GQNMazes(
            cfg.batch_size,
            cfg.num_epochs,
            train=True,
            seq_len=cfg.seq_len,
            data_root=cfg.datadir,
        ).get_batch()
        test_data_batch = GQNMazes(
            cfg.batch_size,
            1,
            train=False,
            seq_len=cfg.eval_seq_len,
            data_root=cfg.datadir,
        ).get_batch()

    else:
        raise ValueError("Dataset {} not supported.".format(cfg.dataset))
    return train_data_batch, test_data_batch


def get_multiple_batches(batch_op, num_batches, sess):
    batches = []
    for _ in range(num_batches):
        batches.append(sess.run(batch_op))
    batches = np.concatenate(batches, 0)
    return batches


def get_single_batch(batch_op, sess):
    return sess.run(batch_op)


import subprocess

if __name__ == "__main__":
    tf.disable_v2_behavior()  # TensorFlow 2.xの動作を無効にします

    # テスト用の設定
    batch_size = 50
    epochs = 1
    seq_len = 100
    data_root = './datadir/'  # Oopsデータセットのパスに置き換えてください
    output_dir = 'saved_videos'  # 保存先ディレクトリ

    # 保存先ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # データセットの初期化
    train_dataset = OopsDataset(
        batch_size=batch_size,
        epochs=epochs,
        train=True,
        seq_len=seq_len,
        data_root=data_root
    )

    # バッチ取得用のオペレーション
    train_data_batch = train_dataset.get_batch()

    # セッションの開始
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            # バッチを取得
            batch = sess.run(train_data_batch)
            print(f"バッチの形状: {batch.shape}")  # 期待される形状: (batch_size, seq_len, 64, 64, 3)

            # バッチ内の各シーケンスを動画として保存
            for i, sequence in enumerate(batch):
                # フレームを [0, 255] の範囲に戻し、uint8型に変換
                frames_uint8 = (sequence * 255).astype(np.uint8)

                # 保存する動画ファイルのパス
                video_filename = os.path.join(output_dir, f"sequence_{i}.mp4")

                # 動画ライターの設定（フォーマット、FPS、フレームサイズ）
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' は一般的なMP4コーデック
                fps = 10.0  # フレームレートを適宜調整
                frame_size = (64, 64)  # フレームサイズ

                # VideoWriterオブジェクトの作成
                out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

                if not out.isOpened():
                    print(f"動画ファイルを開くことができませんでした: {video_filename}")
                    continue

                # 各フレームを動画に書き込む
                for frame in frames_uint8:
                    # OpenCVはBGR形式を使用するため、RGBからBGRに変換
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                # VideoWriterを解放
                out.release()
                print(f"保存しました: {video_filename}")

                # ffmpegで変換
                converted_filename = os.path.join(output_dir, f"sequence_{i}_converted.mp4")
                subprocess.run(['ffmpeg', '-i', video_filename, '-vcodec', 'libx264', '-acodec', 'aac', '-strict', '-2', converted_filename])
                print(f"変換して保存しました: {converted_filename}")

        except Exception as e:
            print(f"バッチの取得エラー: {e}")
