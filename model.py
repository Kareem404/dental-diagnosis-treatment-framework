from torch import nn
from MAE.mae import MaskedAutoencoderLIT
from MAE.vit_dw import ViT
import torch

class CompleteModel(nn.Module):
    def __init__(self, sub_image_model_file, cropped_image_model_file, device="cpu", condition_classes=6, treatment_classes=4, freeze=True):
        super().__init__()
        self.device = device
        num_features = 0

        # applying a masked autoencoder for each channel except the mask channel

        self.sub_image_model = MaskedAutoencoderLIT.load_from_checkpoint(
            sub_image_model_file,
            size="base",
            in_chans=1,
            base_lr=3e-5,
            num_gpus=1,
            batch_size=32,
            warmup_epochs=1,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        ).model.to(device)
        num_features += 768

        self.cropped_image_model = MaskedAutoencoderLIT.load_from_checkpoint(
            cropped_image_model_file,
            size="base",
            in_chans=1,
            base_lr=3e-5,
            num_gpus=1,
            batch_size=32,
            warmup_epochs=1,
            weight_decay=0.05,
            betas=(0.9, 0.95), # what does the betas mean
        ).model.to(device)
        num_features += 768 # the output for each image is a network containing 768 images

        # self.coords_ffn = CoordsFFN(256, 128, dropout_rate=0.5).to(device) # (if we want to use the coordinates as features)
        # num_features += 128

        # applying depth-wise ViT (applies individually to each channel). Need to alter some args
        self.dw_model = ViT(
            image_size=224,
            patch_size=16, 
            num_classes=1,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dim_head=64,         # 768 / 12 = 64
            dropout=0.1,
            emb_dropout=0.1
        ).to(device)
        num_features += 768

        self.condition_classifier = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, condition_classes)
        )

        self.treatment_classifier = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, treatment_classes)
        )

        # freezing the weight of the masked autoencoder because it is already trained. 
        if freeze:
            for param in self.sub_image_model.parameters():
                param.requires_grad = False

            for param in self.cropped_image_model.parameters():
                param.requires_grad = False

    # applying a patched autoencoder for each channel (excluding masked channel) and applying depth-wise ViT to ...
    def forward(self, img_data):
        dw_embed = self.dw_model(img_data) # applies the ViT first to capture features per channel
        cls_sub = self.sub_image_model.forward_encoder(img_data[:,0].unsqueeze(1), 0)[0][:, 0]
        cls_crop = self.cropped_image_model.forward_encoder(img_data[:,1].unsqueeze(1), 0)[0][:, 0]
        # coords_embed = self.coords_ffn(coords_data)
        concatenated = torch.cat([dw_embed, cls_sub, cls_crop], dim=1).to(self.device)
        cond = self.condition_classifier(concatenated) 
        treat = self.treatment_classifier(concatenated)
        return cond, treat