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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path

from g2p_en import G2p
import audio as Audio
from model import PreDefinedEmbedder
from text import grapheme_to_phoneme
from utils.pitch_tools import get_pitch, get_cont_lf0, get_lf0_cwt
from utils.tools import get_phoneme_level_energy, plot_embedding, spec_f0_to_figure


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

        self.with_f0 = preprocess_config["preprocessing"]["pitch"]["with_f0"]
        self.with_f0cwt = preprocess_config["preprocessing"]["pitch"]["with_f0cwt"]
        assert preprocess_config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
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
        # self.STFT = Audio.stft.FastSpeechSTFT(
        #     preprocess_config["preprocessing"]["stft"]["filter_length"],
        #     preprocess_config["preprocessing"]["stft"]["hop_length"],
        #     preprocess_config["preprocessing"]["stft"]["win_length"],
        #     preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        #     preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        #     preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        #     preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        # )
        self.val_unsup_prior = self.val_prior_names(os.path.join(self.out_dir, "val_unsup.txt"))
        self.val_sup_prior = self.val_prior_names(os.path.join(self.out_dir, "val_sup.txt"))
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
        os.makedirs((os.path.join(self.out_dir, "mel_unsup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel_sup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "f0_unsup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "f0_sup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_unsup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_sup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "cwt_spec_unsup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "cwt_spec_sup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "cwt_scales_unsup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "cwt_scales_sup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "f0cwt_mean_std_unsup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "f0cwt_mean_std_sup")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_unsup_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_sup_frame")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_sup_phone")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel2ph")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "attn_prior")), exist_ok=True)
        os.makedirs(embedding_dir, exist_ok=True)

        print("Processing Data ...")
        out_unsup = list()
        out_sup = list()
        filtered_out_unsup = set()
        filtered_out_sup = set()
        train_unsup = list()
        val_unsup = list()
        train_sup = list()
        val_sup = list()
        n_frames = 0
        max_seq_len = -float('inf')
        mel_unsup_min = np.ones(80) * float('inf')
        mel_unsup_max = np.ones(80) * -float('inf')
        mel_sup_min = np.ones(80) * float('inf')
        mel_sup_max = np.ones(80) * -float('inf')
        f0s_unsup = []
        f0s_sup = []
        energy_unsup_frame_scaler = StandardScaler()
        energy_sup_frame_scaler = StandardScaler()
        energy_sup_phone_scaler = StandardScaler()

        def partial_fit(scaler, value):
            if len(value) > 0:
                scaler.partial_fit(value.reshape((-1, 1)))

        def compute_f0_stats(f0s):
            if len(f0s) > 0:
                f0s = np.concatenate(f0s, 0)
                f0s = f0s[f0s != 0]
                f0_mean = np.mean(f0s).item()
                f0_std = np.std(f0s).item()
            return (f0_mean, f0_std)

        def compute_pitch_stats(pitch_scaler, pitch_dir="pitch"):
            if self.pitch_normalization:
                pitch_mean = pitch_scaler.mean_[0]
                pitch_std = pitch_scaler.scale_[0]
            else:
                # A numerical trick to avoid normalization...
                pitch_mean = 0
                pitch_std = 1

            pitch_min, pitch_max = self.normalize(
                os.path.join(self.out_dir, pitch_dir), pitch_mean, pitch_std
            )
            return (pitch_min, pitch_max, pitch_mean, pitch_std)

        def compute_energy_stats(energy_scaler, energy_dir="energy"):
            if self.energy_normalization:
                energy_mean = energy_scaler.mean_[0]
                energy_std = energy_scaler.scale_[0]
            else:
                # A numerical trick to avoid normalization...
                energy_mean = 0
                energy_std = 1

            energy_min, energy_max = self.normalize(
                os.path.join(self.out_dir, energy_dir), energy_mean, energy_std
            )
            return (energy_min, energy_max, energy_mean, energy_std)

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
                        f0_unsup,
                        f0_sup,
                        energy_unsup_frame,
                        energy_sup_frame,
                        energy_sup_phone,
                        n,
                        m_unsup_min,
                        m_unsup_max,
                        m_sup_min,
                        m_sup_max,
                        spker_embed,
                    ) = self.process_utterance(tg_path, speaker, basename, save_speaker_emb)

                    if info_unsup is None and info_sup is None:
                        filtered_out_unsup.add(basename)
                        filtered_out_sup.add(basename)
                        continue
                    else:
                        # Save unsupervised duration features
                        if info_unsup is not None:
                            if self.val_unsup_prior is not None:
                                if basename not in self.val_unsup_prior:
                                    train_unsup.append(info_unsup)
                                else:
                                    val_unsup.append(info_unsup)
                            else:
                                out_unsup.append(info_unsup)

                            if len(f0_unsup) > 0:
                                f0s_unsup.append(f0_unsup)
                            partial_fit(energy_unsup_frame_scaler, energy_unsup_frame)
                        else:
                            filtered_out_unsup.add(basename)
                        # Save sup information
                        if info_sup is not None:
                            if self.val_sup_prior is not None:
                                if basename not in self.val_sup_prior:
                                    train_sup.append(info_sup)
                                else:
                                    val_sup.append(info_sup)
                            else:
                                out_sup.append(info_sup)

                            if len(f0_sup) > 0:
                                f0s_sup.append(f0_sup)
                            partial_fit(energy_sup_frame_scaler, energy_sup_frame)
                            partial_fit(energy_sup_phone_scaler, energy_sup_phone)
                        else:
                            filtered_out_sup.add(basename)

                        if save_speaker_emb:
                            self.speaker_emb_dict[speaker].append(spker_embed)

                        mel_unsup_min = np.minimum(mel_unsup_min, m_unsup_min)
                        mel_unsup_max = np.maximum(mel_unsup_max, m_unsup_max)
                        mel_sup_min = np.minimum(mel_sup_min, m_sup_min)
                        mel_sup_max = np.maximum(mel_sup_max, m_sup_max)

                        if n > max_seq_len:
                            max_seq_len = n

                        n_frames += n

                # Calculate and save mean speaker embedding of this speaker
                if save_speaker_emb:
                    spker_embed_filename = '{}-spker_embed.npy'.format(speaker)
                    np.save(os.path.join(self.out_dir, 'spker_embed', spker_embed_filename), \
                        np.mean(self.speaker_emb_dict[speaker], axis=0), allow_pickle=False)

        print("Computing statistic quantities ...")
        f0s_unsup_stats = compute_f0_stats(f0s_unsup)
        f0s_sup_stats = compute_f0_stats(f0s_sup)

        # Perform normalization if necessary
        energy_unsup_frame_stats = compute_energy_stats(
            energy_unsup_frame_scaler,
            energy_dir="energy_unsup_frame",
        )
        energy_sup_frame_stats = compute_energy_stats(
            energy_sup_frame_scaler,
            energy_dir="energy_sup_frame",
        )
        energy_sup_phone_stats = compute_energy_stats(
            energy_sup_phone_scaler,
            energy_dir="energy_sup_phone",
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "f0_unsup": [float(var) for var in f0s_unsup_stats],
                "f0_sup": [float(var) for var in f0s_sup_stats],
                "energy_unsup_frame": [float(var) for var in energy_unsup_frame_stats],
                "energy_sup_frame": [float(var) for var in energy_sup_frame_stats],
                "energy_sup_phone": [float(var) for var in energy_sup_phone_stats],
                "spec_unsup_min": mel_unsup_min.tolist(),
                "spec_unsup_max": mel_unsup_max.tolist(),
                "spec_sup_min": mel_sup_min.tolist(),
                "spec_sup_max": mel_sup_max.tolist(),
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
        if self.val_unsup_prior is not None:
            assert len(out_unsup) == 0
            random.shuffle(train_unsup)
            train_unsup = [r for r in train_unsup if r is not None]
            val_unsup = [r for r in val_unsup if r is not None]
        else:
            assert len(train_unsup) == 0 and len(val_unsup) == 0
            random.shuffle(out_unsup)
            out_unsup = [r for r in out_unsup if r is not None]
            train_unsup = out_unsup[self.val_size :]
            val_unsup = out_unsup[: self.val_size]

        if self.val_sup_prior is not None:
            assert len(out_sup) == 0
            random.shuffle(train_sup)
            train_sup = [r for r in train_sup if r is not None]
            val_sup = [r for r in val_sup if r is not None]
        else:
            assert len(train_sup) == 0 and len(val_sup) == 0
            random.shuffle(out_sup)
            out_sup = [r for r in out_sup if r is not None]
            train_sup = out_sup[self.val_size :]
            val_sup = out_sup[: self.val_size]

        # Write metadata
        with open(os.path.join(self.out_dir, "train_unsup.txt"), "w", encoding="utf-8") as f:
            for m in train_unsup:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val_unsup.txt"), "w", encoding="utf-8") as f:
            for m in val_unsup:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "train_sup.txt"), "w", encoding="utf-8") as f:
            for m in train_sup:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val_sup.txt"), "w", encoding="utf-8") as f:
            for m in val_sup:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "filtered_out_unsup.txt"), "w", encoding="utf-8") as f:
            for m in sorted(filtered_out_unsup):
                f.write(str(m) + "\n")
        with open(os.path.join(self.out_dir, "filtered_out_sup.txt"), "w", encoding="utf-8") as f:
            for m in sorted(filtered_out_sup):
                f.write(str(m) + "\n")

        return out_unsup, out_sup

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

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        # wav, mel_spectrogram, energy = self.STFT.mel_spectrogram(wav)
        mel_spectrogram = mel_spectrogram[:, : duration]
        energy = energy[: duration]

        # Compute pitch
        if self.with_f0:
            f0_unsup, pitch_unsup = self.get_pitch(wav, mel_spectrogram.T)
            if f0_unsup is None or sum(f0_unsup) == 0:
                unsup_out_exist = False
            else:
                # spec_f0_to_figure(mel_spectrogram.T, {"f0_unsup":f0_unsup}, filename=os.path.join(self.out_dir, f"{basename}_unsup.png"))
                f0_unsup = f0_unsup[: duration]
                pitch_unsup = pitch_unsup[: duration]
                if self.with_f0cwt:
                    cwt_spec_unsup, cwt_scales_unsup, f0cwt_mean_std_unsup = self.get_f0cwt(f0_unsup)
                    if np.any(np.isnan(cwt_spec_unsup)):
                        unsup_out_exist = False
                assert mel_spectrogram.shape[1] == len(f0_unsup)

        if unsup_out_exist:
            # Compute alignment prior
            attn_prior = self.beta_binomial_prior_distribution(
                mel_spectrogram.shape[1],
                len(phone),
                self.beta_binomial_scaling_factor,
            )

            # Frame-level variance
            energy_unsup_frame = copy.deepcopy(energy)

            mel_spectrogram_unsup = copy.deepcopy(mel_spectrogram)

            # Save files
            attn_prior_filename = "{}-attn_prior-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "attn_prior", attn_prior_filename), attn_prior)

            f0_filename = "{}-f0-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "f0_unsup", f0_filename), f0_unsup)

            pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "pitch_unsup", pitch_filename), pitch_unsup)

            cwt_spec_filename = "{}-cwt_spec-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "cwt_spec_unsup", cwt_spec_filename), cwt_spec_unsup)

            cwt_scales_filename = "{}-cwt_scales-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "cwt_scales_unsup", cwt_scales_filename), cwt_scales_unsup)

            f0cwt_mean_std_filename = "{}-f0cwt_mean_std-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "f0cwt_mean_std_unsup", f0cwt_mean_std_filename), f0cwt_mean_std_unsup)

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
            phone, duration, mel2ph, start, end = self.get_alignment(
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

                # Compute mel-scale spectrogram and energy
                mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
                # wav, mel_spectrogram, energy = self.STFT.mel_spectrogram(wav)
                mel_spectrogram = mel_spectrogram[:, : sum(duration)]
                energy = energy[: sum(duration)]

                # Compute pitch
                if self.with_f0:
                    f0_sup, pitch_sup = self.get_pitch(wav, mel_spectrogram.T)
                    if f0_sup is None or sum(f0_unsup) == 0:
                        sup_out_exist = False
                    else:
                        # spec_f0_to_figure(mel_spectrogram.T, {"f0_sup":f0_sup}, filename=os.path.join(self.out_dir, f"{basename}_sup.png"))
                        f0_sup = f0_sup[: sum(duration)]
                        pitch_sup = pitch_sup[: sum(duration)]
                        if self.with_f0cwt:
                            cwt_spec_sup, cwt_scales_sup, f0cwt_mean_std_sup = self.get_f0cwt(f0_sup)
                            if np.any(np.isnan(cwt_spec_sup)):
                                sup_out_exist = False
                        assert mel_spectrogram.shape[1] == len(f0_sup)

                if sup_out_exist:
                    # Frame-level variance
                    energy_sup_frame = copy.deepcopy(energy)

                    # Phone-level variance
                    energy_sup_phone = get_phoneme_level_energy(duration, energy)

                    mel_spectrogram_sup = copy.deepcopy(mel_spectrogram)

                    # Save files
                    dur_filename = "{}-duration-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

                    mel2ph_filename = "{}-mel2ph-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "mel2ph", mel2ph_filename), mel2ph)

                    f0_filename = "{}-f0-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "f0_sup", f0_filename), f0_sup)

                    pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "pitch_sup", pitch_filename), pitch_sup)

                    cwt_spec_filename = "{}-cwt_spec-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "cwt_spec_sup", cwt_spec_filename), cwt_spec_sup)

                    cwt_scales_filename = "{}-cwt_scales-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "cwt_scales_sup", cwt_scales_filename), cwt_scales_sup)

                    f0cwt_mean_std_filename = "{}-f0cwt_mean_std-{}.npy".format(speaker, basename)
                    np.save(os.path.join(self.out_dir, "f0cwt_mean_std_sup", f0cwt_mean_std_filename), f0cwt_mean_std_sup)

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
            return tuple([None]*13)
        else:
            return (
                "|".join([basename, speaker, text_unsup, raw_text]) if unsup_out_exist else None,
                "|".join([basename, speaker, text_sup, raw_text]) if sup_out_exist else None,
                f0_unsup if unsup_out_exist else None,
                f0_sup if sup_out_exist else None,
                self.remove_outlier(energy_unsup_frame) if unsup_out_exist else None,
                self.remove_outlier(energy_sup_frame) if sup_out_exist else None,
                self.remove_outlier(energy_sup_phone) if sup_out_exist else None,
                max(mel_spectrogram_unsup.shape[1] if unsup_out_exist else -1, mel_spectrogram_sup.shape[1] if sup_out_exist else -1),
                np.min(mel_spectrogram_unsup, axis=1) if unsup_out_exist else np.ones(80) * float('inf'),
                np.max(mel_spectrogram_unsup, axis=1) if unsup_out_exist else np.ones(80) * -float('inf'),
                np.min(mel_spectrogram_sup, axis=1) if sup_out_exist else np.ones(80) * float('inf'),
                np.max(mel_spectrogram_sup, axis=1) if sup_out_exist else np.ones(80) * -float('inf'),
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
        mel2ph = []
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

        # Get mel2ph
        for ph_idx in range(len(phones)):
            mel2ph += [ph_idx + 1] * durations[ph_idx]
        assert sum(durations) == len(mel2ph)

        return phones, durations, mel2ph, start_time, end_time

    def get_pitch(self, wav, mel):
        f0, pitch_coarse = get_pitch(wav, mel, self.preprocess_config)
        return f0, pitch_coarse

    def get_f0cwt(self, f0):
        uv, cont_lf0_lpf = get_cont_lf0(f0)
        logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
        logf0s_mean_std_org = np.array([logf0s_mean_org, logf0s_std_org])
        cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        return Wavelet_lf0, scales, logf0s_mean_std_org

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
