import os
import re
import random
import json
import copy

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.stats import betabinom
# from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path

from g2p_en import G2p
import audio as Audio
from model import PreDefinedEmbedder
from text import grapheme_to_phoneme
from utils.tools import get_phoneme_level_pitch, get_phoneme_level_energy, plot_embedding


class Preprocessor:
    def __init__(self, preprocess_config, model_config, train_config):
        random.seed(train_config['seed'])
        self.preprocess_config = preprocess_config
        self.multi_speaker = model_config["multi_speaker"]
        # self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        self.corpus_dir = preprocess_config["path"]["corpus_path"]
        self.in_dir = preprocess_config["path"]["raw_path"]
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        self.val_size = preprocess_config["preprocessing"]["val_size"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.filter_length = preprocess_config["preprocessing"]["stft"]["filter_length"]
        self.trim_top_db = preprocess_config["preprocessing"]["audio"]["trim_top_db"]
        self.beta_binomial_scaling_factor = preprocess_config["preprocessing"]["duration"]["beta_binomial_scaling_factor"]

        assert preprocess_config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert preprocess_config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]

        self.pitch_normalization = preprocess_config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = preprocess_config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
        self.val_sup_prior = self.val_prior_names(os.path.join(self.out_dir, "val_sup.txt"))
        self.val_unsup_prior = self.val_prior_names(os.path.join(self.out_dir, "val_unsup.txt"))
        self.speaker_emb = None
        self.in_sub_dirs = [p for p in os.listdir(self.in_dir) if os.path.isdir(os.path.join(self.in_dir, p))]
        if self.multi_speaker and preprocess_config["preprocessing"]["speaker_embedder"] != "none":
            self.speaker_emb = PreDefinedEmbedder(preprocess_config)
            self.speaker_emb_dict = self._init_spker_embeds(self.in_sub_dirs)
        self.g2p = G2p()

    def _init_spker_embeds(self, spkers):
        spker_embeds = dict()
        for spker in spkers:
            spker_embeds[spker] = list()
        return spker_embeds

    def val_prior_names(self, val_prior_path):
        val_prior_names = set()
        if os.path.isfile(val_prior_path):
            print("Load pre-defined validation set...")
            with open(val_prior_path, "r", encoding="utf-8") as f:
                for m in f.readlines():
                    val_prior_names.add(m.split("|")[0])
            return list(val_prior_names)
        else:
            return None

    def build_from_path(self):
        embedding_dir = os.path.join(self.out_dir, "spker_embed")
        os.makedirs((os.path.join(self.out_dir, "mel_sup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel_unsup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_unsup_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_sup_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_sup_phone")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_unsup_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_sup_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_sup_phone")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "attn_prior")), exist_ok=True)
        os.makedirs(embedding_dir, exist_ok=True)

        print("Processing Data ...")
        out = list()
        filtered_out_unsup = set()
        filtered_out_sup = set()
        train_unsup = list()
        val_unsup = list()
        train_sup = list()
        val_sup = list()
        n_frames = 0
        max_seq_len = -float('inf')
        pitch_unsup_frame_scaler = StandardScaler()
        pitch_sup_frame_scaler = StandardScaler()
        pitch_sup_phone_scaler = StandardScaler()
        energy_unsup_frame_scaler = StandardScaler()
        energy_sup_frame_scaler = StandardScaler()
        energy_sup_phone_scaler = StandardScaler()

        def partial_fit(scaler, value):
            if len(value) > 0:
                scaler.partial_fit(value.reshape((-1, 1)))

        def compute_stats(pitch_scaler, energy_scaler, pitch_dir="pitch", energy_dir="energy"):
            if self.pitch_normalization:
                pitch_mean = pitch_scaler.mean_[0]
                pitch_std = pitch_scaler.scale_[0]
            else:
                # A numerical trick to avoid normalization...
                pitch_mean = 0
                pitch_std = 1
            if self.energy_normalization:
                energy_mean = energy_scaler.mean_[0]
                energy_std = energy_scaler.scale_[0]
            else:
                energy_mean = 0
                energy_std = 1

            pitch_min, pitch_max = self.normalize(
                os.path.join(self.out_dir, pitch_dir), pitch_mean, pitch_std
            )
            energy_min, energy_max = self.normalize(
                os.path.join(self.out_dir, energy_dir), energy_mean, energy_std
            )
            return (pitch_min, pitch_max, pitch_mean, pitch_std), (energy_min, energy_max, energy_mean, energy_std)

        skip_speakers = set()
        for embedding_name in os.listdir(embedding_dir):
            skip_speakers.add(embedding_name.split("-")[0])

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(self.in_sub_dirs)):
            save_speaker_emb = self.speaker_emb is not None and speaker not in skip_speakers
            if os.path.isdir(os.path.join(self.in_dir, speaker)):
                speakers[speaker] = i
                for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                    if ".wav" not in wav_name:
                        continue

                    basename = wav_name.split(".")[0]
                    tg_path = os.path.join(
                        self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                    )
                    (
                        info_unsup,
                        info_sup,
                        pitch_unsup_frame,
                        pitch_sup_frame,
                        pitch_sup_phone,
                        energy_unsup_frame,
                        energy_sup_frame,
                        energy_sup_phone,
                        n,
                        spker_embed,
                    ) = self.process_utterance(tg_path, speaker, basename, save_speaker_emb)

                    if info_unsup is None and info_sup is None:
                        filtered_out_unsup.add(basename)
                        filtered_out_sup.add(basename)
                        continue
                    else:
                        # Save unsupervised duration features
                        if info_unsup is not None:
                            if self.val_sup_prior is not None:
                                if basename not in self.val_sup_prior:
                                    train_unsup.append(info_unsup)
                                else:
                                    val_unsup.append(info_unsup)
                            else:
                                out.append(info_unsup)

                            partial_fit(pitch_unsup_frame_scaler, pitch_unsup_frame)
                            partial_fit(energy_unsup_frame_scaler, energy_unsup_frame)
                        else:
                            filtered_out_unsup.add(basename)
                        # Save sup information
                        if info_sup is not None:
                            if self.val_unsup_prior is not None:
                                if basename not in self.val_unsup_prior:
                                    train_sup.append(info_sup)
                                else:
                                    val_sup.append(info_sup)
                            else:
                                out.append(info_sup)

                            partial_fit(pitch_sup_frame_scaler, pitch_sup_frame)
                            partial_fit(pitch_sup_phone_scaler, pitch_sup_phone)
                            partial_fit(energy_sup_frame_scaler, energy_sup_frame)
                            partial_fit(energy_sup_phone_scaler, energy_sup_phone)
                        else:
                            filtered_out_sup.add(basename)

                        if save_speaker_emb:
                            self.speaker_emb_dict[speaker].append(spker_embed)

                        if n > max_seq_len:
                            max_seq_len = n

                        n_frames += n

                # Calculate and save mean speaker embedding of this speaker
                assert len(self.speaker_emb_dict[speaker]) > 0, "embedding of {} is empty!".format(speaker)
                if save_speaker_emb:
                    spker_embed_filename = '{}-spker_embed.npy'.format(speaker)
                    np.save(os.path.join(self.out_dir, 'spker_embed', spker_embed_filename), \
                        np.mean(self.speaker_emb_dict[speaker], axis=0), allow_pickle=False)

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        pitch_unsup_frame_stats, energy_unsup_frame_stats = compute_stats(
            pitch_unsup_frame_scaler,
            energy_unsup_frame_scaler,
            pitch_dir="pitch_unsup_frame",
            energy_dir="energy_unsup_frame",
        )
        pitch_sup_frame_stats, energy_sup_frame_stats = compute_stats(
            pitch_sup_frame_scaler,
            energy_sup_frame_scaler,
            pitch_dir="pitch_sup_frame",
            energy_dir="energy_sup_frame",
        )
        pitch_sup_phone_stats, energy_sup_phone_stats = compute_stats(
            pitch_sup_phone_scaler,
            energy_sup_phone_scaler,
            pitch_dir="pitch_sup_phone",
            energy_dir="energy_sup_phone",
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch_unsup_frame": [float(var) for var in pitch_unsup_frame_stats],
                "pitch_sup_frame": [float(var) for var in pitch_sup_frame_stats],
                "pitch_sup_phone": [float(var) for var in pitch_sup_phone_stats],
                "energy_unsup_frame": [float(var) for var in energy_unsup_frame_stats],
                "energy_sup_frame": [float(var) for var in energy_sup_frame_stats],
                "energy_sup_phone": [float(var) for var in energy_sup_phone_stats],
                "max_seq_len": max_seq_len,
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        if self.speaker_emb is not None:
            print("Plot speaker embedding...")
            plot_embedding(
                self.out_dir, *self.load_embedding(embedding_dir),
                self.divide_speaker_by_gender(self.corpus_dir), filename="spker_embed_tsne.png"
            )

        # Save dataset
        filtered_out_unsup, filtered_out_sup = list(filtered_out_unsup), list(filtered_out_sup)
        if self.val_sup_prior is not None:
            assert len(out) == 0
            random.shuffle(train_sup)
            train_sup = [r for r in train_sup if r is not None]
            val_sup = [r for r in val_sup if r is not None]
        else:
            assert len(train_sup) == 0 and len(val_sup) == 0
            random.shuffle(out)
            out = [r for r in out if r is not None]
            train_sup = out[self.val_size :]
            val_sup = out[: self.val_size]

        if self.val_unsup_prior is not None:
            assert len(out) == 0
            random.shuffle(train_unsup)
            train_unsup = [r for r in train_unsup if r is not None]
            val_unsup = [r for r in val_unsup if r is not None]
        else:
            assert len(train_unsup) == 0 and len(val_unsup) == 0
            random.shuffle(out)
            out = [r for r in out if r is not None]
            train_unsup = out[self.val_size :]
            val_unsup = out[: self.val_size]

        # Write metadata
        with open(os.path.join(self.out_dir, "train_sup.txt"), "w", encoding="utf-8") as f:
            for m in train_sup:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val_sup.txt"), "w", encoding="utf-8") as f:
            for m in val_sup:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "train_unsup.txt"), "w", encoding="utf-8") as f:
            for m in train_unsup:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val_unsup.txt"), "w", encoding="utf-8") as f:
            for m in val_unsup:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "filtered_out_unsup.txt"), "w", encoding="utf-8") as f:
            for m in sorted(filtered_out_unsup):
                f.write(str(m) + "\n")
        with open(os.path.join(self.out_dir, "filtered_out_sup.txt"), "w", encoding="utf-8") as f:
            for m in sorted(filtered_out_sup):
                f.write(str(m) + "\n")

        return out

    def load_audio(self, wav_path):
        wav_raw, _ = librosa.load(wav_path, self.sampling_rate)
        _, index = librosa.effects.trim(wav_raw, top_db=self.trim_top_db, frame_length=self.filter_length, hop_length=self.hop_length)
        wav = wav_raw[index[0]:index[1]]
        duration = (index[1] - index[0]) / self.hop_length
        return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)

    def process_utterance(self, tg_path, speaker, basename, save_speaker_emb):
        sup_out_exist, unsup_out_exist = True, True
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))

        wav_raw, wav, duration = self.load_audio(wav_path)
        spker_embed = self.speaker_emb(wav) if save_speaker_emb else None

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")
        phone = grapheme_to_phoneme(raw_text, self.g2p)
        phones = "{" + "}{".join(phone) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        text_unsup = phones.replace("}{", " ")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = pitch[: duration]
        if np.sum(pitch != 0) <= 1:
            unsup_out_exist = False
        else:
            # Compute mel-scale spectrogram and energy
            mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
            mel_spectrogram = mel_spectrogram[:, : duration]
            energy = energy[: duration]

            # Compute alignment prior
            attn_prior = self.beta_binomial_prior_distribution(
                mel_spectrogram.shape[1],
                len(phone),
                self.beta_binomial_scaling_factor,
            )

            # Frame-level variance
            pitch_unsup_frame, energy_unsup_frame = copy.deepcopy(pitch), copy.deepcopy(energy)

            mel_spectrogram_unsup = copy.deepcopy(mel_spectrogram)

            # Save files
            attn_prior_filename = "{}-attn_prior-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "attn_prior", attn_prior_filename), attn_prior)

            pitch_frame_filename = "{}-pitch-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "pitch_unsup_frame", pitch_frame_filename), pitch_unsup_frame)

            energy_frame_filename = "{}-energy-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "energy_unsup_frame", energy_frame_filename), energy_unsup_frame)

            mel_unsup_filename = "{}-mel-{}.npy".format(speaker, basename)
            np.save(
                os.path.join(self.out_dir, "mel_unsup", mel_unsup_filename),
                mel_spectrogram_unsup.T,
            )

        # Supervised duration features
        if os.path.exists(tg_path):
            # Get alignments
            textgrid = tgt.io.read_textgrid(tg_path)
            phone, duration, start, end = self.get_alignment(
                textgrid.get_tier_by_name("phones")
            )
            text_sup = "{" + " ".join(phone) + "}"
            if start >= end:
                sup_out_exist = False
            else:
                # Read and trim wav files
                wav, _ = librosa.load(wav_path, self.sampling_rate)
                wav = wav.astype(np.float32)
                wav = wav[
                    int(self.sampling_rate * start) : int(self.sampling_rate * end)
                ]

                # Compute fundamental frequency
                pitch, t = pw.dio(
                    wav.astype(np.float64),
                    self.sampling_rate,
                    frame_period=self.hop_length / self.sampling_rate * 1000,
                )
                pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

                pitch = pitch[: sum(duration)]
                if np.sum(pitch != 0) <= 1:
                    sup_out_exist = False
                else:
                    # Compute mel-scale spectrogram and energy
                    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
                    mel_spectrogram = mel_spectrogram[:, : sum(duration)]
                    energy = energy[: sum(duration)]

                    # Frame-level variance
                    pitch_sup_frame, energy_sup_frame = copy.deepcopy(pitch), copy.deepcopy(energy)

                    # Phone-level variance
                    pitch_sup_phone, energy_sup_phone = get_phoneme_level_pitch(duration, pitch), get_phoneme_level_energy(duration, energy)

                    mel_spectrogram_sup = copy.deepcopy(mel_spectrogram)

                    # Save files
                    dur_filename = "{}-duration-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

                    pitch_frame_filename = "{}-pitch-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "pitch_sup_frame", pitch_frame_filename), pitch_sup_frame)

                    pitch_phone_filename = "{}-pitch-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "pitch_sup_phone", pitch_phone_filename), pitch_sup_phone)

                    energy_frame_filename = "{}-energy-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "energy_sup_frame", energy_frame_filename), energy_sup_frame)

                    energy_phone_filename = "{}-energy-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "energy_sup_phone", energy_phone_filename), energy_sup_phone)

                    mel_sup_filename = "{}-mel-{}.npy".format(speaker, basename)
                    np.save(
                        os.path.join(self.out_dir, "mel_sup", mel_sup_filename),
                        mel_spectrogram_sup.T,
                    )
        else:
            sup_out_exist = False

        if not sup_out_exist and not unsup_out_exist:
            return tuple([None]*10)
        else:
            return (
                "|".join([basename, speaker, text_unsup, raw_text]) if unsup_out_exist else None,
                "|".join([basename, speaker, text_sup, raw_text]) if sup_out_exist else None,
                self.remove_outlier(pitch_unsup_frame) if unsup_out_exist else None,
                self.remove_outlier(pitch_sup_frame) if sup_out_exist else None,
                self.remove_outlier(pitch_sup_phone) if sup_out_exist else None,
                self.remove_outlier(energy_unsup_frame) if unsup_out_exist else None,
                self.remove_outlier(energy_sup_frame) if sup_out_exist else None,
                self.remove_outlier(energy_sup_phone) if sup_out_exist else None,
                max(mel_spectrogram_unsup.shape[1] if unsup_out_exist else -1, mel_spectrogram_sup.shape[1] if sup_out_exist else -1),
                spker_embed,
            )

    def beta_binomial_prior_distribution(self, phoneme_count, mel_count, scaling_factor=1.0):
        P, M = phoneme_count, mel_count
        x = np.arange(0, P)
        mel_text_probs = []
        for i in range(1, M+1):
            a, b = scaling_factor*i, scaling_factor*(M+1-i)
            rv = betabinom(P, a, b)
            mel_i_prob = rv.pmf(x)
            mel_text_probs.append(mel_i_prob)
        return np.array(mel_text_probs)

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

    def divide_speaker_by_gender(self, in_dir, speaker_path="speaker-info.txt"):
        speakers = dict()
        with open(os.path.join(in_dir, speaker_path), encoding='utf-8') as f:
            for line in tqdm(f):
                if "ID" in line: continue
                parts = [p.strip() for p in re.sub(' +', ' ',(line.strip())).split(' ')]
                spk_id, gender = parts[0], parts[2]
                speakers[str(spk_id)] = gender
        return speakers

    def load_embedding(self, embedding_dir):
        embedding_path_list = [_ for _ in Path(embedding_dir).rglob('*.npy')]
        embedding = None
        embedding_speaker_id = list()
        # Gather data
        for path in tqdm(embedding_path_list):
            embedding = np.concatenate((embedding, np.load(path)), axis=0) \
                                            if embedding is not None else np.load(path)
            embedding_speaker_id.append(str(str(path).split('/')[-1].split('-')[0]))
        return embedding, embedding_speaker_id
