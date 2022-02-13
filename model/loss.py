import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.tools import get_variance_level, ssim
from text import sil_phonemes_ids


class CompTransTTSLoss(nn.Module):
    """ CompTransTTS Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(CompTransTTSLoss, self).__init__()
        _, self.energy_feature_level = \
                get_variance_level(preprocess_config, model_config, data_loading=False)
        self.loss_config = train_config["loss"]
        self.pitch_config = preprocess_config["preprocessing"]["pitch"]
        self.pitch_type = self.pitch_config["pitch_type"]
        self.use_pitch_embed = model_config["variance_embedding"]["use_pitch_embed"]
        self.use_energy_embed = model_config["variance_embedding"]["use_energy_embed"]
        self.model_type = model_config["prosody_modeling"]["model_type"]
        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        self.binarization_loss_enable_steps = train_config["duration"]["binarization_loss_enable_steps"]
        self.binarization_loss_warmup_steps = train_config["duration"]["binarization_loss_warmup_steps"]
        self.gmm_mdn_beta = train_config["prosody"]["gmm_mdn_beta"]
        self.prosody_loss_enable_steps = train_config["prosody"]["prosody_loss_enable_steps"]
        self.var_start_steps = train_config["step"]["var_start_steps"]
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.sil_ph_ids = sil_phonemes_ids()

    # def gaussian_probability(self, sigma, mu, target, mask=None):
    #     target = target.unsqueeze(2).expand_as(sigma)
    #     ret = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    #     if mask is not None:
    #         ret = ret.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
    #     return torch.prod(ret, 3)

    # def mdn_loss(self, w, sigma, mu, target, mask=None):
    #     """
    #     w -- [B, src_len, num_gaussians]
    #     sigma -- [B, src_len, num_gaussians, out_features]
    #     mu -- [B, src_len, num_gaussians, out_features]
    #     target -- [B, src_len, out_features]
    #     mask -- [B, src_len]
    #     """
    #     prob = w * self.gaussian_probability(sigma, mu, target, mask)
    #     nll = -torch.log(torch.sum(prob, dim=2))
    #     if mask is not None:
    #         nll = nll.masked_fill(mask, 0)
    #     l_pp = torch.sum(nll, dim=1)
    #     return torch.mean(l_pp)

    def log_gaussian_probability(self, sigma, mu, target, mask=None):
        """
        prob -- [B, src_len, num_gaussians]
        """
        target = target.unsqueeze(2).expand_as(sigma)
        prob = torch.log((1.0 / (math.sqrt(2 * math.pi)*sigma))) -0.5 * ((target - mu) / sigma)**2
        if mask is not None:
            prob = prob.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
        prob = torch.sum(prob, dim=3)

        return prob

    def mdn_loss(self, w, sigma, mu, target, mask=None):
        """
        w -- [B, src_len, num_gaussians]
        sigma -- [B, src_len, num_gaussians, out_features]
        mu -- [B, src_len, num_gaussians, out_features]
        target -- [B, src_len, out_features]
        mask -- [B, src_len]
        """
        prob = torch.log(w) + self.log_gaussian_probability(sigma, mu, target, mask)
        nll = -torch.logsumexp(prob, 2)
        if mask is not None:
            nll = nll.masked_fill(mask, 0)
        return torch.mean(nll)

        # prob = w * self.gaussian_probability(sigma, mu, target, mask)
        # # prob = w.unsqueeze(-1) * self.gaussian_probability(sigma, mu, target, mask)
        # nll = -torch.log(torch.sum(prob, dim=2))
        # # nll = -torch.sum(prob, dim=2)
        # if mask is not None:
        #     nll = nll.masked_fill(mask, 0)
        #     # nll = nll.masked_fill(mask.unsqueeze(-1), 0)
        # l_pp = torch.sum(nll, dim=1)
        # return torch.mean(l_pp)

    def get_mel_loss(self, mel_predictions, mel_targets):
        mel_targets.requires_grad = False
        mel_predictions = mel_predictions.masked_fill(self.mel_masks_fill.unsqueeze(-1), 0)
        mel_targets = mel_targets.masked_fill(self.mel_masks_fill.unsqueeze(-1), 0)
        mel_loss = self.l1_loss(mel_predictions, mel_targets)
        return mel_loss

    def l1_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        l1_loss = F.l1_loss(decoder_output, target, reduction="none")
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def ssim_loss(self, decoder_output, target, bias=6.0):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        weights = self.weights_nonzero_speech(target)
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        ssim_loss = (ssim_loss * weights).sum() / weights.sum()
        return ssim_loss

    def weights_nonzero_speech(self, target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

    def get_duration_loss(self, dur_pred, dur_gt, txt_tokens):
        """
        :param dur_pred: [B, T], float, log scale
        :param txt_tokens: [B, T]
        :return:
        """
        dur_gt.requires_grad = False
        losses = {}
        B, T = txt_tokens.shape
        nonpadding = self.src_masks.float()
        dur_gt = dur_gt.float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p_id in self.sil_ph_ids:
            is_sil = is_sil | (txt_tokens == p_id)
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if self.loss_config["dur_loss"] == "mse":
            losses["pdur"] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction="none")
            losses["pdur"] = (losses["pdur"] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        elif self.loss_config["dur_loss"] == "mog":
            return NotImplementedError
        elif self.loss_config["dur_loss"] == "crf":
            # losses["pdur"] = -self.model.dur_predictor.crf(
            #     dur_pred, dur_gt.long().clamp(min=0, max=31), mask=nonpadding > 0, reduction="mean")
            return NotImplementedError
        losses["pdur"] = losses["pdur"] * self.loss_config["lambda_ph_dur"]

        # use linear scale for sent and word duration
        if self.loss_config["lambda_word_dur"] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction="none")
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses["wdur"] = wdur_loss * self.loss_config["lambda_word_dur"]
        if self.loss_config["lambda_sent_dur"] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction="mean")
            losses["sdur"] = sdur_loss.mean() * self.loss_config["lambda_sent_dur"]
        return losses

    def get_pitch_loss(self, pitch_predictions, pitch_targets):
        for _, pitch_target in pitch_targets.items():
            if pitch_target is not None:
                pitch_target.requires_grad = False
        losses = {}
        if self.pitch_type == "ph":
            nonpadding = self.src_masks.float()
            pitch_loss_fn = F.l1_loss if self.loss_config["pitch_loss"] == "l1" else F.mse_loss
            losses["f0"] = (pitch_loss_fn(pitch_predictions["pitch_pred"][:, :, 0], pitch_targets["f0"],
                                          reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_f0"]
        else:
            mel2ph = pitch_targets["mel2ph"]  # [B, T_s]
            f0 = pitch_targets["f0"]
            uv = pitch_targets["uv"]
            nonpadding = self.mel_masks.float()
            if self.pitch_type == "cwt":
                cwt_spec = pitch_targets[f"cwt_spec"]
                f0_mean = pitch_targets["f0_mean"]
                f0_std = pitch_targets["f0_std"]
                cwt_pred = pitch_predictions["cwt"][:, :, :10]
                f0_mean_pred = pitch_predictions["f0_mean"]
                f0_std_pred = pitch_predictions["f0_std"]
                losses["C"] = self.cwt_loss(cwt_pred, cwt_spec) * self.loss_config["lambda_f0"]
                if self.pitch_config["use_uv"]:
                    assert pitch_predictions["cwt"].shape[-1] == 11
                    uv_pred = pitch_predictions["cwt"][:, :, -1]
                    losses["uv"] = (F.binary_cross_entropy_with_logits(uv_pred, uv, reduction="none") * nonpadding) \
                                    .sum() / nonpadding.sum() * self.loss_config["lambda_uv"]
                losses["f0_mean"] = F.l1_loss(f0_mean_pred, f0_mean) * self.loss_config["lambda_f0"]
                losses["f0_std"] = F.l1_loss(f0_std_pred, f0_std) * self.loss_config["lambda_f0"]
                # if self.loss_config["cwt_add_f0_loss"]:
                #     f0_cwt_ = cwt2f0_norm(cwt_pred, f0_mean_pred, f0_std_pred, mel2ph, self.pitch_config)
                #     self.add_f0_loss(f0_cwt_[:, :, None], f0, uv, losses, nonpadding=nonpadding)
            elif self.pitch_type == "frame":
                self.add_f0_loss(pitch_predictions["pitch_pred"], f0, uv, losses, nonpadding=nonpadding)
        return losses

    def add_f0_loss(self, p_pred, f0, uv, losses, nonpadding):
        assert p_pred[..., 0].shape == f0.shape
        if self.pitch_config["use_uv"]:
            assert p_pred[..., 1].shape == uv.shape
            losses["uv"] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_uv"]
            nonpadding = nonpadding * (uv == 0).float()

        f0_pred = p_pred[:, :, 0]
        if self.loss_config["pitch_loss"] in ["l1", "l2"]:
            pitch_loss_fn = F.l1_loss if self.loss_config["pitch_loss"] == "l1" else F.mse_loss
            losses["f0"] = (pitch_loss_fn(f0_pred, f0, reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_f0"]
        elif self.loss_config["pitch_loss"] == "ssim":
            return NotImplementedError

    def cwt_loss(self, cwt_p, cwt_g):
        if self.loss_config["cwt_loss"] == "l1":
            return F.l1_loss(cwt_p, cwt_g)
        if self.loss_config["cwt_loss"] == "l2":
            return F.mse_loss(cwt_p, cwt_g)
        if self.loss_config["cwt_loss"] == "ssim":
            return self.ssim_loss(cwt_p, cwt_g, 20)

    def get_energy_loss(self, energy_predictions, energy_targets):
        energy_targets.requires_grad = False
        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(self.src_masks)
            energy_targets = energy_targets.masked_select(self.src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(self.mel_masks)
            energy_targets = energy_targets.masked_select(self.mel_masks)
        energy_loss = F.l1_loss(energy_predictions, energy_targets)
        return energy_loss

    def get_init_losses(self, device):
        duration_loss = {
            "pdur": torch.zeros(1).to(device),
            "wdur": torch.zeros(1).to(device),
            "sdur": torch.zeros(1).to(device),
        }
        pitch_loss = {}
        if self.pitch_type == "ph":
            pitch_loss["f0"] = torch.zeros(1).to(device)
        else:
            if self.pitch_type == "cwt":
                pitch_loss["C"] = torch.zeros(1).to(device)
                if self.pitch_config["use_uv"]:
                    pitch_loss["uv"] = torch.zeros(1).to(device)
                pitch_loss["f0_mean"] = torch.zeros(1).to(device)
                pitch_loss["f0_std"] = torch.zeros(1).to(device)
            elif self.pitch_type == "frame":
                if self.pitch_config["use_uv"]:
                    pitch_loss["uv"] = torch.zeros(1).to(device)
                if self.loss_config["pitch_loss"] in ["l1", "l2"]:
                    pitch_loss["f0"] = torch.zeros(1).to(device)
        energy_loss = torch.zeros(1).to(device)
        return duration_loss, pitch_loss, energy_loss

    def forward(self, inputs, predictions, step):
        (
            texts,
            _,
            _,
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            _,
            _,
        ) = inputs[3:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs,
            prosody_info,
        ) = predictions
        self.src_masks = ~src_masks
        mel_masks = ~mel_masks
        if self.learn_alignment:
            attn_soft, attn_hard, attn_hard_dur, attn_logprob = attn_outs
            duration_targets = attn_hard_dur
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        self.mel_masks = mel_masks[:, :mel_masks.shape[1]]
        self.mel_masks_fill = ~self.mel_masks

        mel_loss = self.get_mel_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.get_mel_loss(postnet_mel_predictions, mel_targets)

        ctc_loss = bin_loss = torch.zeros(1).to(mel_targets.device)
        if self.learn_alignment:
            ctc_loss = self.sum_loss(attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens)
            if step < self.binarization_loss_enable_steps:
                bin_loss_weight = 0.
            else:
                bin_loss_weight = min((step-self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
            bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight

        prosody_loss = torch.zeros(1).to(mel_targets.device)
        if self.training and self.model_type == "du2021" and step > self.prosody_loss_enable_steps:
            w, sigma, mu, prosody_embeddings = prosody_info
            prosody_loss = self.gmm_mdn_beta * self.mdn_loss(w, sigma, mu, prosody_embeddings.detach(), ~src_masks)
        elif self.training and self.model_type == "liu2021" and step > self.prosody_loss_enable_steps:
            up_tgt, pp_tgt, up_vec, pp_vec, _ = prosody_info
            prosody_loss = F.l1_loss(up_tgt, up_vec)
            # prosody_loss = F.l1_loss(
            prosody_loss += F.l1_loss(
                pp_tgt.masked_select(src_masks.unsqueeze(-1)), pp_vec.masked_select(src_masks.unsqueeze(-1)))

        total_loss = mel_loss + postnet_mel_loss + ctc_loss + bin_loss + prosody_loss

        duration_loss, pitch_loss, energy_loss = self.get_init_losses(mel_targets.device)
        if step > self.var_start_steps:
            duration_loss = self.get_duration_loss(log_duration_predictions, duration_targets, texts)
            if self.use_pitch_embed:
                pitch_loss = self.get_pitch_loss(pitch_predictions, pitch_targets)
            if self.use_energy_embed:
                energy_loss = self.get_energy_loss(energy_predictions, energy_targets)
            total_loss += sum(duration_loss.values()) + sum(pitch_loss.values()) + energy_loss

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            ctc_loss,
            bin_loss,
            prosody_loss,
        )


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()
