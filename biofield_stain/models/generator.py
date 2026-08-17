"""
SPADEUNetGenerator: H&E → IHC translation generator.

SPADE-UNet conditioned on UNI pathology features + HER2 class embedding.
Encoder processes H&E input, decoder uses SPADE conditioning from UNI features
+ FiLM from class embedding, with skip connections.

~30M params at 512, supports 1024 with extra encoder/decoder levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from biofield_stain.models.blocks import SPADEBlock, ResBlock, SelfAttention
from biofield_stain.models.edge_encoder import EdgeEncoder, MultiScaleEdgeEncoder
from biofield_stain.models.uni_processor import UNIFeatureProcessor, UNIFeatureProcessorHighRes


class MarkerSpecificExpertBank(nn.Module):
    """Lightweight residual expert bank for marker-specific decoder features."""

    def __init__(self, channels, num_classes, hidden_dim=32, residual_scale=0.10,
                 adaptive_gate=False, gate_hidden_dim=16, gate_init=0.50):
        super().__init__()
        self.num_classes = num_classes
        self.residual_scale = residual_scale
        self.adaptive_gate = adaptive_gate
        hidden = min(hidden_dim, channels)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1),
            )
            for _ in range(num_classes)
        ])
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)
        if adaptive_gate:
            gate_hidden = min(gate_hidden_dim, channels)
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channels, gate_hidden, kernel_size=1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(gate_hidden, 1, kernel_size=1),
                )
                for _ in range(num_classes)
            ])
            gate_init = max(1e-4, min(1.0 - 1e-4, float(gate_init)))
            gate_bias = torch.logit(torch.tensor(gate_init)).item()
            for gate in self.gates:
                nn.init.zeros_(gate[-1].weight)
                nn.init.constant_(gate[-1].bias, gate_bias)
        else:
            self.gates = None

    def forward(self, x, labels):
        if self.residual_scale <= 0:
            return x
        out = x
        safe_labels = labels.clamp(min=0, max=self.num_classes - 1)
        for label_idx in safe_labels.unique().tolist():
            if int(label_idx) == self.num_classes - 1:
                continue
            mask = safe_labels == int(label_idx)
            if not torch.any(mask):
                continue
            residual = self.experts[int(label_idx)](x[mask])
            if self.gates is not None:
                gate = torch.sigmoid(self.gates[int(label_idx)](x[mask]))
                residual = residual * gate
            if out is x:
                out = x.clone()
            out[mask] = out[mask] + self.residual_scale * residual
        return out


class BioFieldDecoderModulator(nn.Module):
    """Spatial BioField FiLM modulation for decoder features.

    A marker-specific expression field is converted into a bounded residual
    feature modulation. The last projection is zero-initialized so enabling
    the module starts from the original generator behaviour and learns into
    the new path safely.
    """

    def __init__(self, channels, num_classes, hidden_dim=64, residual_scale=0.10):
        super().__init__()
        self.num_classes = num_classes
        self.residual_scale = residual_scale
        hidden = min(hidden_dim, max(8, channels // 2))
        groups = 8 if hidden % 8 == 0 else 4
        self.net = nn.Sequential(
            nn.Conv2d(1 + num_classes, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, biofield, labels):
        if biofield is None or self.residual_scale <= 0:
            return x
        b, _, h, w = x.shape
        safe_labels = labels.clamp(min=0, max=self.num_classes - 1)
        field = biofield.float()
        if field.shape[-2:] != (h, w):
            field = F.interpolate(field, size=(h, w), mode='bilinear', align_corners=False)
        one_hot = F.one_hot(safe_labels, num_classes=self.num_classes).float()
        cond = one_hot[:, :, None, None].expand(b, self.num_classes, h, w)
        cond = cond.to(device=x.device, dtype=field.dtype)
        gamma, beta = self.net(torch.cat([field.to(x.device), cond], dim=1)).chunk(2, dim=1)
        active = (safe_labels != self.num_classes - 1).to(dtype=x.dtype, device=x.device)
        active = active[:, None, None, None]
        residual = x.float() * torch.tanh(gamma) + beta
        residual = residual.to(dtype=x.dtype) * active
        return x + self.residual_scale * residual


class SPADEUNetGenerator(nn.Module):
    """SPADE-UNet generator for H&E → HER2 translation.

    Encoder processes H&E input into multi-scale features.
    Decoder uses SPADE conditioning from UNI features + FiLM from class embedding.
    Skip connections from encoder to decoder.

    ~30M params.
    """

    def __init__(self, num_classes=5, class_dim=64, uni_dim=1024,
                 input_skip=False, edge_encoder=False, edge_base_ch=32,
                 uni_spatial_size=4, image_size=512, uni_spade_at_512=False,
                 stain_experts=False, stain_expert_hidden=32,
                 stain_expert_scale=0.10, stain_expert_layers=(4, 3, 2),
                 stain_expert_adaptive_gate=False,
                 stain_expert_gate_hidden=16,
                 stain_expert_gate_init=0.50,
                 biofield_decoder_mod=False,
                 biofield_decoder_hidden=64,
                 biofield_decoder_scale=0.10,
                 biofield_decoder_layers=(5, 4, 3, 2)):
        super().__init__()
        self.num_classes = num_classes
        self.class_dim = class_dim
        self.input_skip = input_skip
        self.edge_encoder_flag = edge_encoder
        self.uni_spatial_size = uni_spatial_size
        self.image_size = image_size
        self.uni_spade_at_512 = uni_spade_at_512
        self.stain_expert_layers = set(int(l) for l in stain_expert_layers)
        self.biofield_decoder_layers = set(int(l) for l in biofield_decoder_layers)

        # Class embedding (5 classes: 0, 1+, 2+, 3+, null)
        self.class_embed = nn.Embedding(num_classes, class_dim)

        # UNI feature processor — choose based on spatial resolution
        if uni_spatial_size >= 16:
            # High-res patch tokens (e.g., 32x32 = 1024 tokens)
            self.uni_processor = UNIFeatureProcessorHighRes(
                uni_dim=uni_dim, base_channels=512, spatial_size=uni_spatial_size,
                output_512=(uni_spade_at_512 and image_size == 1024),
            )
        else:
            # Original CLS-token features (4x4 = 16 tokens)
            self.uni_processor = UNIFeatureProcessor(
                uni_dim=uni_dim, base_channels=512,
            )

        # Edge encoder (parallel structure pathway)
        # Note: edge encoder always operates at 512 resolution.
        # For 1024 input, H&E is downsampled to 512 before edge extraction.
        self.edge_encoder_type = edge_encoder  # False, 'v1', or 'v2'
        if edge_encoder == 'v2':
            self.edge_encoder = MultiScaleEdgeEncoder(base_ch=edge_base_ch)
            edge_ch = {512: edge_base_ch, 256: edge_base_ch, 128: edge_base_ch * 2,
                       64: edge_base_ch * 4, 32: edge_base_ch * 4}
        elif edge_encoder:  # True or 'v1'
            self.edge_encoder = EdgeEncoder(base_ch=edge_base_ch)
            edge_ch = {512: 0, 256: edge_base_ch, 128: edge_base_ch * 2,
                       64: edge_base_ch * 4, 32: edge_base_ch * 4}
        else:
            self.edge_encoder = None
            edge_ch = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0}

        # === 1024 support: extra encoder/decoder levels ===
        if image_size == 1024:
            # enc0: 1024→512 (lightweight, just spatial downsample)
            self.enc0 = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            enc1_in_ch = 32  # enc1 takes enc0 output, not raw H&E
        else:
            self.enc0 = None
            enc1_in_ch = 3  # enc1 takes raw H&E at 512

        # Encoder
        self.enc1 = nn.Sequential(  # 512→256
            nn.Conv2d(enc1_in_ch, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(  # 256→128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(  # 128→64
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc4 = nn.Sequential(  # 64→32
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc5 = nn.Sequential(  # 32→16
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck (at 16×16)
        self.bottleneck = nn.Sequential(
            ResBlock(512),
            SelfAttention(512),
            ResBlock(512),
        )

        # Decoder with SPADE conditioning
        # Channel counts: main_skip + edge_skip (if enabled) + upsampled
        # D5: 512 (up) + 512 (skip e4) + edge_ch[32] → 512
        self.dec5_conv = nn.Conv2d(512 + 512 + edge_ch[32], 512, 3, padding=1)
        self.dec5_spade = SPADEBlock(512, uni_channels=512, class_dim=class_dim)
        self.dec5_act = nn.LeakyReLU(0.2, inplace=True)

        # D4: 512 (up) + 256 (skip e3) + edge_ch[64] → 256
        self.dec4_conv = nn.Conv2d(512 + 256 + edge_ch[64], 256, 3, padding=1)
        self.dec4_spade = SPADEBlock(256, uni_channels=256, class_dim=class_dim)
        self.dec4_act = nn.LeakyReLU(0.2, inplace=True)

        # D3: 256 (up) + 128 (skip e2) + edge_ch[128] → 128
        self.dec3_conv = nn.Conv2d(256 + 128 + edge_ch[128], 128, 3, padding=1)
        self.dec3_spade = SPADEBlock(128, uni_channels=128, class_dim=class_dim)
        self.dec3_act = nn.LeakyReLU(0.2, inplace=True)

        # D2: 128 (up) + 64 (skip e1) + edge_ch[256] → 64
        self.dec2_conv = nn.Conv2d(128 + 64 + edge_ch[256], 64, 3, padding=1)
        self.dec2_spade = SPADEBlock(64, uni_channels=64, class_dim=class_dim)
        self.dec2_act = nn.LeakyReLU(0.2, inplace=True)

        if image_size == 1024:
            # D1 (new): upsample 256→512, skip from enc0 (32ch) + edge@512
            dec1_in_ch = 64 + 32 + edge_ch[512]
            if uni_spade_at_512:
                # UNI SPADE conditioning at 512 level (uni_ch=32 at this scale)
                self.dec1_conv = nn.Conv2d(dec1_in_ch, 64, 3, padding=1)
                self.dec1_spade = SPADEBlock(64, uni_channels=32, class_dim=class_dim)
                self.dec1_act = nn.LeakyReLU(0.2, inplace=True)
            else:
                self.dec1_conv = nn.Sequential(
                    nn.Conv2d(dec1_in_ch, 64, 3, padding=1),
                    nn.InstanceNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                self.dec1_spade = None
                self.dec1_act = None
            # Output: upsample 512→1024, optional H&E input skip
            output_in_ch = 64 + (3 if input_skip else 0)
            self.output = nn.Sequential(
                nn.Conv2d(output_in_ch, 64, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Tanh(),
            )
        else:
            self.dec1_conv = None
            # Output: concat H&E input (3ch if input_skip) + edge@512 (if v2)
            output_in_ch = 64 + (3 if input_skip else 0) + edge_ch[512]
            self.output = nn.Sequential(
                nn.Conv2d(output_in_ch, 64, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Tanh(),
            )

        if stain_experts:
            self.stain_experts = nn.ModuleDict()
            expert_channels = {5: 512, 4: 256, 3: 128, 2: 64, 1: 64}
            for layer_idx in sorted(self.stain_expert_layers):
                if layer_idx not in expert_channels:
                    raise ValueError(
                        f"Unsupported stain expert layer {layer_idx}; "
                        "choose from 5,4,3,2,1"
                    )
                self.stain_experts[str(layer_idx)] = MarkerSpecificExpertBank(
                    channels=expert_channels[layer_idx],
                    num_classes=num_classes,
                    hidden_dim=stain_expert_hidden,
                    residual_scale=stain_expert_scale,
                    adaptive_gate=stain_expert_adaptive_gate,
                    gate_hidden_dim=stain_expert_gate_hidden,
                    gate_init=stain_expert_gate_init,
                )
        else:
            self.stain_experts = None

        if biofield_decoder_mod:
            self.biofield_modulators = nn.ModuleDict()
            mod_channels = {5: 512, 4: 256, 3: 128, 2: 64, 1: 64}
            for layer_idx in sorted(self.biofield_decoder_layers):
                if layer_idx not in mod_channels:
                    raise ValueError(
                        f"Unsupported BioField decoder layer {layer_idx}; "
                        "choose from 5,4,3,2,1"
                    )
                self.biofield_modulators[str(layer_idx)] = BioFieldDecoderModulator(
                    channels=mod_channels[layer_idx],
                    num_classes=num_classes,
                    hidden_dim=biofield_decoder_hidden,
                    residual_scale=biofield_decoder_scale,
                )
        else:
            self.biofield_modulators = None

    def _apply_stain_expert(self, layer_idx, x, labels):
        if self.stain_experts is None:
            return x
        key = str(layer_idx)
        if key not in self.stain_experts:
            return x
        expert = self.stain_experts[key]
        return expert(x, labels)

    def _apply_biofield_modulator(self, layer_idx, x, biofield_condition, labels):
        if self.biofield_modulators is None:
            return x
        key = str(layer_idx)
        if key not in self.biofield_modulators:
            return x
        return self.biofield_modulators[key](x, biofield_condition, labels)

    def encode(self, images):
        """Extract intermediate encoder features for PatchNCE loss.

        Args:
            images: [B, 3, H, H] in [-1, 1] (H&E or generated IHC)

        Returns:
            dict mapping layer index to feature tensor:
                {1: [B, 64, 256, 256], 2: [B, 128, 128, 128],
                 3: [B, 256, 64, 64], 4: [B, 512, 32, 32]}
        """
        if self.enc0 is not None:
            e0 = self.enc0(images)
            enc1_input = e0
        else:
            enc1_input = images

        e1 = self.enc1(enc1_input)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return {1: e1, 2: e2, 3: e3, 4: e4}

    def forward(self, he_images, uni_features, labels, biofield_condition=None):
        """
        Args:
            he_images: [B, 3, H, H] in [-1, 1] where H=512 or H=1024
            uni_features: [B, N, 1024] where N=16 (4x4 CLS) or N=1024 (32x32 patch)
            labels: [B] int class labels (0-4)
            biofield_condition: optional [B, 1, h, w] marker expression prior

        Returns:
            output: [B, 3, H, H] in [-1, 1]
        """
        class_emb = self.class_embed(labels)
        uni_maps = self.uni_processor(uni_features)

        # Edge encoder (parallel structure pathway)
        # Edge encoder always operates at 512 resolution
        if self.edge_encoder_type:
            if self.image_size == 1024:
                he_512 = F.interpolate(he_images, size=512, mode='bilinear', align_corners=False)
                edge_maps = self.edge_encoder(he_512)
            else:
                edge_maps = self.edge_encoder(he_images)
        else:
            edge_maps = None

        # === 1024: extra encoder level ===
        if self.enc0 is not None:
            e0 = self.enc0(he_images)   # [B, 32, 512, 512]
            enc1_input = e0
        else:
            e0 = None
            enc1_input = he_images

        # Encoder
        e1 = self.enc1(enc1_input)  # [B, 64, 256, 256]
        e2 = self.enc2(e1)          # [B, 128, 128, 128]
        e3 = self.enc3(e2)          # [B, 256, 64, 64]
        e4 = self.enc4(e3)          # [B, 512, 32, 32]
        e5 = self.enc5(e4)          # [B, 512, 16, 16]

        # Bottleneck at 16×16
        x = self.bottleneck(e5)     # [B, 512, 16, 16]

        # D5: upsample 16→32, skip from e4 + edge@32, UNI at 32
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip5 = [x, e4] + ([edge_maps[32]] if edge_maps else [])
        x = torch.cat(skip5, dim=1)
        x = self.dec5_conv(x)
        x = self.dec5_spade(x, uni_maps[32], class_emb)
        x = self.dec5_act(x)
        x = self._apply_biofield_modulator(5, x, biofield_condition, labels)
        x = self._apply_stain_expert(5, x, labels)

        # D4: upsample 32→64, skip from e3 + edge@64, UNI at 64
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip4 = [x, e3] + ([edge_maps[64]] if edge_maps else [])
        x = torch.cat(skip4, dim=1)
        x = self.dec4_conv(x)
        x = self.dec4_spade(x, uni_maps[64], class_emb)
        x = self.dec4_act(x)
        x = self._apply_biofield_modulator(4, x, biofield_condition, labels)
        x = self._apply_stain_expert(4, x, labels)

        # D3: upsample 64→128, skip from e2 + edge@128, UNI at 128
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip3 = [x, e2] + ([edge_maps[128]] if edge_maps else [])
        x = torch.cat(skip3, dim=1)
        x = self.dec3_conv(x)
        x = self.dec3_spade(x, uni_maps[128], class_emb)
        x = self.dec3_act(x)
        x = self._apply_biofield_modulator(3, x, biofield_condition, labels)
        x = self._apply_stain_expert(3, x, labels)

        # D2: upsample 128→256, skip from e1 + edge@256, UNI at 256
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip2 = [x, e1] + ([edge_maps[256]] if edge_maps else [])
        x = torch.cat(skip2, dim=1)
        x = self.dec2_conv(x)
        x = self.dec2_spade(x, uni_maps[256], class_emb)
        x = self.dec2_act(x)
        x = self._apply_biofield_modulator(2, x, biofield_condition, labels)
        x = self._apply_stain_expert(2, x, labels)

        if self.image_size == 1024:
            # D1: upsample 256→512, skip from e0 (32ch) + edge@512
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            skip1 = [x, e0] + ([edge_maps[512]] if edge_maps else [])
            x = torch.cat(skip1, dim=1)
            x = self.dec1_conv(x)
            if self.dec1_spade is not None:
                x = self.dec1_spade(x, uni_maps[512], class_emb)
                x = self.dec1_act(x)
            x = self._apply_biofield_modulator(1, x, biofield_condition, labels)
            x = self._apply_stain_expert(1, x, labels)
            # [B, 64, 512, 512]

            # Output: upsample 512→1024, optional H&E input skip
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if self.input_skip:
                x = torch.cat([x, he_images], dim=1)
            x = self.output(x)  # [B, 3, 1024, 1024]
        else:
            # D1: upsample 256→512, optional skip from H&E input + edge@512
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = self._apply_biofield_modulator(1, x, biofield_condition, labels)
            x = self._apply_stain_expert(1, x, labels)
            skip1 = [x]
            if self.input_skip:
                skip1.append(he_images)
            if edge_maps and 512 in edge_maps:
                skip1.append(edge_maps[512])
            x = torch.cat(skip1, dim=1) if len(skip1) > 1 else x
            x = self.output(x)  # [B, 3, 512, 512]

        return x
