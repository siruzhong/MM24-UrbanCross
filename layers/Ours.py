import math
import open_clip_mine as open_clip
import numpy as np
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.init
from torch.nn.parallel.distributed import DistributedDataParallel
import copy
import torch
import torch.nn as nn
import torch.nn.init
from .resnet import resnet50
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


class UrbanCross(nn.Module):
    def __init__(self, args):
        """
        Initialize the UrbanCross model.

        Args:
            args: Model configuration arguments.
        """
        super().__init__()
        # Create OpenCLIP model and transforms
        self.clip_model, _, transform = open_clip.create_model_and_transforms(
            model_name="ViT-L-14",  # coca_ViT-L-14
            pretrained="laion2B-s32B-b82K",  # mscoco_finetuned_laion2B-s13B-b90k
            output_dict=True,
        )
        # Create a copy of the OpenCLIP model for segmented images
        self.clip_img_seg = copy.deepcopy(self.clip_model)
        # Remove the transformer layer from the copied model
        del self.clip_img_seg.transformer

    def forward(self, img, text, segment_imgs):
        """
        Forward pass of the UrbanCross model.

        Args:
            img (torch.Tensor): Input image tensor.
            text (torch.Tensor): Input text tensor.
            segment_imgs (torch.Tensor): Input segmented images tensor.

        Returns:
            torch.Tensor: Similarity scores between image and text.
            torch.Tensor: Similarity scores between segmented images and text.
        """
        with torch.cuda.amp.autocast():
            # Get features for the input image and text
            clip_model_out = self.clip_model(img, text)
            img_emb = clip_model_out["image_features"]
            text_emb = clip_model_out["text_features"]

            # Get the number of segments
            num_seg = segment_imgs.shape[0]
            seg_emb_list = []
            
            # Flatten the segment_imgs tensor
            bs, num_seg, _, _, _ = segment_imgs.shape
            segment_imgs_reshaped = segment_imgs.view(bs * num_seg, 3, 224, 224)
            
            # Encode segmented images
            img_seg_emb = self.clip_img_seg.encode_image(segment_imgs_reshaped)
            img_seg_emb = img_seg_emb.view(bs, num_seg, -1)
            # Calculate the feature mean of each batch
            img_seg_emb = img_seg_emb.mean(dim=1)

            # Calculate cosine similarity between image and text
            sim_img2text = cosine_sim(img_emb, text_emb)
            # Calculate cosine similarity between segmented images and text
            sim_seg2text = cosine_sim(img_seg_emb, text_emb)

        return sim_img2text, sim_seg2text


class AdversarialLoss(nn.Module):
    def __init__(self):
        """
        Initialize the AdversarialLoss module.
        """
        super(AdversarialLoss, self).__init__()
        self.W_tilde_2 = 1.0

    def forward(self, model, F_s_tilde, F_t_tilde, W2):
        """
        Forward pass of the AdversarialLoss module.

        Args:
            model (nn.Module): Discriminator model.
            F_s_tilde (torch.Tensor): Source features tensor.
            F_t_tilde (torch.Tensor): Target features tensor.
            W2 (torch.Tensor): Weight tensor.

        Returns:
            torch.Tensor: Adversarial loss value.
        """
        # Calculate the discriminator's probability on source features
        prob_source = model(F_s_tilde)
        # Calculate the discriminator's probability on target features
        prob_target = model(F_t_tilde)

        # Ensure the discriminator output is in the range [0, 1] by applying sigmoid
        prob_source = torch.sigmoid(prob_source)
        prob_target = torch.sigmoid(prob_target)

        # Expand the weight tensor to match the shape of probability tensors
        W2 = W2.unsqueeze(dim=1)
        
        # Calculate the adversarial loss using weighted cross-entropy loss
        loss = -(
            torch.mean(W2 * torch.log(prob_source)) +
            torch.mean(W2 * torch.log(1 - prob_target))
        )
        
        # Note: Negative sign is used because we typically minimize the loss, and the original equation is for maximization
        return loss


class UrbanCross_finetune(nn.Module):
    def __init__(self, args, word2idx):
        """
        Initialize the UrbanCross_finetune model.

        Args:
            args: Model configuration arguments.
            word2idx: Mapping from words to indices.
        """
        super().__init__()
        # Create OpenCLIP model and transforms
        self.clip_model, _, transform = open_clip.create_model_and_transforms(
            model_name="ViT-L-14",  # CLIP model name
            pretrained="laion2B-s32B-b82K",  # Pretrained weights
            output_dict=True,
        )
        # Create a copy of the OpenCLIP model for segmented images
        self.clip_img_seg = copy.deepcopy(self.clip_model)
        # Remove the transformer layer from the copied model
        del self.clip_img_seg.transformer
       
        # Define the discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 2),  # Output layer with 2 units for binary classification
        )

        # Initialize the adversarial loss module
        self.adv_loss = AdversarialLoss()

        # Initialize the CLIP loss module
        self.clip_loss = open_clip.ClipLoss()

    def forward(self, img, text, segment_imgs):
        """
        Forward pass of the UrbanCross_finetune model.

        Args:
            img (torch.Tensor): Input image tensor.
            text (torch.Tensor): Input text tensor.
            segment_imgs (torch.Tensor): Input segmented images tensor.

        Returns:
            torch.Tensor: Discriminator output.
        """
        # Calculate CLIP model output for source and target images
        clip_model_out_source = self.clip_model(img_source, text_source)
        clip_model_out_target = self.clip_model(img_target, text_target)

        # Extract image embeddings for source and target images
        img_emb_source = clip_model_out_source["image_features"]
        img_emb_target = clip_model_out_target["image_features"]

        # Extract text embeddings for source and target texts
        text_emb_source = clip_model_out_source["text_features"]
        text_emb_target = clip_model_out_target["text_features"]

        # Calculate similarity between text embeddings
        W1 = cosine_sim(text_emb_target, text_emb_source)
        W1_mean = W1.mean(dim=0)

        # Determine the batch size
        batchsize = img_emb_source.shape[0]

        # Select a subset of the batch based on the similarity scores (W1) between text embeddings.
        # The selected batch size is half of the original batch size.
        selected_batchsize = int(batchsize / 2)
        
        # Sort the similarity scores (W1) along each row in descending order to get the top similarities.
        sorted_W1, _ = torch.sort(W1, dim=1, descending=True)
        
        # Select the top similarities for each sample in the batch.
        W2 = sorted_W1[:, :selected_batchsize]
        
        # Sort the mean similarity scores (W1_mean) across the batch in descending order to select the most relevant samples.
        _, sorted_W1_mean_index = torch.sort(W1_mean, descending=True)
        
        # Select the corresponding image and text embeddings for the selected samples based on mean similarity.
        img_emb_source_filtered = img_emb_source[sorted_W1_mean_index[:selected_batchsize]]
        text_emb_source_filtered = text_emb_source[sorted_W1_mean_index[:selected_batchsize]]
        
        # Sum the top similarity scores (W2) over the second dimension to get a vector.
        W2 = torch.sum(W2, dim=1)

        # Normalize the similarity scores (W2) to range [0, 1].
        W2_min = torch.min(W2)
        W2_max = torch.max(W2)
        W2 = (W2 - W2_min) / (W2_max - W2_min)
        W2 = W2 / torch.sum(W2)
        
        # Calculate adversarial loss
        adv_loss = self.adv_loss(
            self.discriminator, img_emb_source_filtered, img_emb_target, W2
        )
        
        # Calculate CLIP loss
        clip_loss = self.clip_loss(
            img_emb_source_filtered,
            text_emb_source_filtered,
            logit_scale=1.0,
        )
        
        loss = clip_loss + adv_loss
        return loss


class ImageExtractFeature(nn.Module):
    def __init__(self, args):
        """
        Initialize the ImageExtractFeature module.

        Args:
            args: Model configuration arguments.
        """
        super(ImageExtractFeature, self).__init__()
        self.embed_dim = args.embed_dim
        self.is_finetune = args.is_finetune

        # Load pre-trained ResNet50 model
        self.resnet = resnet50(args, num_classes=30, pretrained=True)

        # Vision Multi-Scale Fusion Module
        self.vmsf = VMSF(args)

        # Filtering local features
        self.conv_filter = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=32, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )

        # Set requires_grad parameter for ResNet parameters based on finetune flag
        for param in self.resnet.parameters():
            param.requires_grad = self.is_finetune

    def forward(self, img):
        """
        Forward pass of the ImageExtractFeature module.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            tuple: Tuple containing shallow features and fused deep features.
        """
        # Shallow features
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # Deep features
        deep_fea_1 = self.resnet.layer2(self.resnet.layer1(x))
        deep_fea_2 = self.resnet.layer3(deep_fea_1)
        deep_fea_3 = self.resnet.layer4(deep_fea_2)

        # Apply convolutional filter for shallow features
        shallow_fea = self.conv_filter(x)

        # Combine deep features using Vision Multi-Scale Fusion Module
        deep_feas = (deep_fea_1, deep_fea_2, deep_fea_3)
        vg_emb = self.vmsf(deep_feas)
        
        return shallow_fea, vg_emb


class TextExtractFeature(nn.Module):
    def __init__(self, args, word2idx):
        """
        Initialize the TextExtractFeature module.

        Args:
            args: Model configuration arguments.
            word2idx (dict): Mapping from words to their indices.
        """
        super(TextExtractFeature, self).__init__()
        self.gpuid = args.gpuid
        self.embed_dim = args.embed_dim
        self.word_dim = args.word_dim
        self.vocab_size = 8590
        self.num_layers = args.num_layers
        self.use_bidirectional_rnn = args.use_bidirectional_rnn
        
        # Word embedding layer
        self.embed = nn.Embedding(self.vocab_size, self.word_dim)

        # caption embedding
        self.rnn = nn.GRU(
            self.word_dim,
            self.embed_dim,
            self.num_layers,
            batch_first=True,
            bidirectional=self.use_bidirectional_rnn,
        )
        
        # Linear projection layer
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

        # Dropout layer
        self.dropout = nn.Dropout(0.4)

        # Initialize word embeddings with pre-trained GloVe vectors
        self.init_weights(word2idx, self.word_dim)

    def init_weights(self, word2idx, word_dim):
        """
        Initialize word embeddings with pre-trained GloVe vectors.

        Args:
            word2idx (dict): Mapping from words to their indices.
            word_dim (int): Dimension of word embeddings.
        """
        # Load pretrained GloVe word embeddings
        wemb = torchtext.vocab.GloVe()

        assert wemb.vectors.shape[1] == word_dim

        # Initialize word embeddings with GloVe vectors
        missing_words = []
        for word, idx in word2idx.items():
            if word not in wemb.stoi:
                word = word.replace("-", "").replace(".", "").replace("'", "")
                if "/" in word:
                    word = word.split("/")[0]
            if word in wemb.stoi:
                self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
            else:
                missing_words.append(word)
        print(
            "Words: {}/{} found in vocabulary; {} words missing".format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)
            )
        )

    def forward(self, x, lengths):
        """
        Forward pass of the TextExtractFeature module.

        Args:
            x (torch.Tensor): Input word indices tensor.
            lengths (list): List of sequence lengths.

        Returns:
            torch.Tensor: Extracted text features.
        """
        # Embed word indices to vectors
        x = self.dropout(self.embed(x))
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # Forward propagate through RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        # If bidirectional, average forward and backward hidden states
        if self.use_bidirectional_rnn:
            cap_emb = (
                cap_emb[:, :, : int(cap_emb.size(2) / 2)]
                + cap_emb[:, :, int(cap_emb.size(2) / 2) :]
            ) / 2

        return cap_emb


class VMSF(nn.Module):
    def __init__(self, args):
        """
        Vision Multi-Scale Fusion module for combining features from different depths.

        Args:
            args: Model configuration arguments.
        """
        super(VMSF, self).__init__()
        self.embed_dim = args.embed_dim # Dimensionality of the embedded features
        self.dropout_r = 0.2    # Dropout rate
        self.use_relu = True    # Whether to use ReLU activation

        # Convolutional layers for feature transformation
        self.conv_512 = nn.Conv2d(
            in_channels=512, out_channels=self.embed_dim, kernel_size=1, stride=1
        )
        self.conv_1024 = nn.Conv2d(
            in_channels=1024, out_channels=self.embed_dim, kernel_size=1, stride=1
        )
        self.conv_2048 = nn.Conv2d(
            in_channels=2048, out_channels=self.embed_dim * 2, kernel_size=1, stride=1
        )

        # Upsampling layers for adjusting feature spatial dimensions
        self.up_sample_double = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_sample_half = nn.Upsample(scale_factor=0.5, mode="nearest")

        # Channel attention mechanism for emphasizing important features
        self.channel_att = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embed_dim * 4,
                out_channels=self.embed_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                self.embed_dim,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(inplace=True),
        )

        # MLP-based channel filter for further feature refinement
        self.channel_filter = MLP(
            self.embed_dim,
            self.embed_dim * 2,
            self.embed_dim,
            self.dropout_r,
            self.use_relu,
        )

    def forward(self, deep_feas):
        """
        Forward pass of the VMSF module.

        Args:
            deep_feas (tuple): Tuple containing features from different depths.

        Returns:
            torch.Tensor: High-level fused feature representation.
        """
        d1, d2, d3 = deep_feas  # Unpack the input features

        # Apply convolution and upsampling operations
        p_2 = self.conv_1024(d2)
        up_4 = self.up_sample_double(self.conv_2048(d3))
        up_2 = self.up_sample_half(self.conv_512(d1))

        # Depth concatenation and channel attention
        ms_fea = self.channel_att(torch.cat([up_2, p_2, up_4], dim=1))

        # Mean pooling and channel filtering
        high_emb = self.channel_filter(ms_fea.mean(-1).mean(-1))

        return high_emb


class SFGS(nn.Module):
    def __init__(self, args, dim=32):
        """
        Scene Fine-Grained Sensing Module for processing visual features.

        Args:
            args: Model configuration arguments.
            dim (int): Dimensionality of the visual features. Defaults to 32.
        """
        super(SFGS, self).__init__()
        self.args = args
        self.embed_dim = args.embed_dim
        self.dim = dim
        self.dropout_r = 0.1
        self.use_relu = True

        # 1x1 convolution block followed by adaptive average pooling
        self.conv2d_block_11 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
        )
        
        # 3x3 convolution block followed by adaptive average pooling
        self.conv2d_block_33 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=3, bias=False
            ),
            nn.BatchNorm2d(
                32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
        )

        # 5x5 convolution block followed by adaptive average pooling
        self.conv2d_block_55 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5, stride=5, bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
        )

        # Fully connected layer
        self.fc = FC(self.embed_dim // 2, self.embed_dim, self.dropout_r, self.use_relu)

        # Weighted Self-Attention module
        self.wsa = WSA(args, num_dim=128, is_weighted=True)

    def forward(self, vl_fea):
        """
        Forward pass of the SFGS module.

        Args:
            vl_fea (torch.Tensor): Input visual features.

        Returns:
            torch.Tensor: Output feature representation.
        """
        bs, dim, _, _ = vl_fea.size()

        # Apply different convolutional blocks and reshape
        vl_1 = self.conv2d_block_11(vl_fea).view(bs, dim, -1)
        vl_2 = self.conv2d_block_33(vl_fea).view(bs, dim, -1)
        vl_3 = self.conv2d_block_55(vl_fea).view(bs, dim * 2, -1)

        # Concatenate the feature maps
        vl_depth = torch.cat([vl_1, vl_2, vl_3], dim=1)

        # Apply fully connected layer followed by weighted self-attention
        return self.wsa(self.fc(vl_depth)).mean(1)


class Aggregation(nn.Module):
    def __init__(self, args):
        """
        Aggregation module for combining visual and global embeddings.

        Args:
            args: Model configuration arguments.
        """
        super(Aggregation, self).__init__()
        self.args = args
        self.embed_dim = args.embed_dim

        # Fully connected layers for aggregation
        self.fc_1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.fc_2 = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, vl_emb, gl_emb):
        """
        Forward pass of the Aggregation module.

        Args:
            vl_emb (torch.Tensor): Visual embeddings.
            gl_emb (torch.Tensor): Global embeddings.

        Returns:
            torch.Tensor: Aggregated feature representation.
        """
        # Depth concat of visual and global embeddings
        v_emb = torch.cat([vl_emb, gl_emb], dim=1)

        # Apply fully connected layers and activation function
        return self.fc_2(torch.relu(self.fc_1(v_emb)))


class TCGE(nn.Module):
    def __init__(self, args):
        """
        Text Coarse-Grained Enhancement module for enhancing textual features.

        Args:
            args: Model configuration arguments.
        """
        super(TCGE, self).__init__()
        self.embed_dim = args.embed_dim
        self.gpuid = args.gpuid

        # Batch normalization layer
        self.bn_1d = nn.BatchNorm1d(self.embed_dim)
        
        # Graph Attention module
        self.ga = GA(args)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

        # Multi-layer perceptron (MLP)
        self.mlp = MLP(self.embed_dim, self.embed_dim * 2, self.embed_dim, 0.1, True)

        # 1D convolutional blocks
        self.conv1d_block_22 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                stride=2,
                kernel_size=2,
            ),
            nn.BatchNorm1d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.conv1d_block_33 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                stride=3,
                kernel_size=3,
            ),
            nn.BatchNorm1d(
                512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, cap_emb, lengths):
        """
        Forward pass of the TCGE module.

        Args:
            cap_emb (torch.Tensor): Caption embeddings.
            lengths (list): Lengths of the input captions.

        Returns:
            torch.Tensor: Enhanced textual embeddings.
        """
        # Graph Attention (GA) Embeddings
        bs, dim, emb_dim = cap_emb.size()
        ga_emb = cap_emb + self.dropout(
            self.bn_1d(self.ga(cap_emb).view(bs * dim, -1)).view(bs, dim, -1)
        )
        
        # Joint Word Embeddings (JW)
        f2 = self.conv1d_block_22(cap_emb.permute(0, 2, 1)).permute(0, 2, 1)
        f3 = self.conv1d_block_33(cap_emb.permute(0, 2, 1)).permute(0, 2, 1)
        jw_emb = torch.cat([f2, f3], dim=1)

        # GA-JW Fusion
        ga_jw = torch.cat([ga_emb, jw_emb], dim=1)
        tex_emb = self.mlp(ga_jw) + ga_jw

        # Length-based pooling
        I = torch.LongTensor(lengths).view(-1, 1, 1)  # 100, 1, 1
        I = Variable(I.expand(tex_emb.size(0), 1, self.embed_dim) - 1).cuda(
            self.gpuid
        ) 
        out = torch.gather(tex_emb, 1, I).squeeze(1)

        return l2norm(out, dim=-1)


class MHAtt(nn.Module):
    def __init__(self, args):
        """
        Multi-Head Attention module.

        Args:
            args: Model configuration arguments.
        """
        super(MHAtt, self).__init__()
        self.embed_dim = args.embed_dim
        self.dropout_r = 0.1
        
        # Linear transformations for keys, queries, and values
        self.linear_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Linear transformation for merging attention heads
        self.linear_merge = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_r)

    def forward(self, v, k, q, mask=None):
        """
        Forward pass of the Multi-Head Attention module.

        Args:
            v (torch.Tensor): Values.
            k (torch.Tensor): Keys.
            q (torch.Tensor): Queries.
            mask (torch.Tensor, optional): Masking tensor.

        Returns:
            torch.Tensor: Attended output.
        """
        bs = q.size(0)

        # Linear transformations followed by reshaping for multi-head attention
        v = self.linear_v(v).view(bs, -1, 8, 64).transpose(1, 2)
        k = self.linear_k(k).view(bs, -1, 8, 64).transpose(1, 2)
        q = self.linear_q(q).view(bs, -1, 8, 64).transpose(1, 2)

        # Calculate attention scores and apply dropout
        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(bs, -1, self.embed_dim)

        # Linear transformation after attention
        atted = self.linear_merge(atted)

        return atted

    def att(self, k, q, v, mask=None):
        """
        Calculate attention scores.

        Args:
            k (torch.Tensor): Keys.
            q (torch.Tensor): Queries.
            v (torch.Tensor): Values.
            mask (torch.Tensor, optional): Masking tensor.

        Returns:
            torch.Tensor: Attended output.
        """
        d_k = q.shape[-1]

        # Calculate attention scores using matrix multiplication and scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply masking if provided
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        # Apply softmax to obtain attention weights and apply dropout
        att_map = torch.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        # Calculate attended output
        return torch.matmul(att_map, v)


class WSA(nn.Module):
    def __init__(self, args, num_dim=128, is_weighted=False):
        """
        Word-Sentence Attention module.

        Args:
            args: Model configuration arguments.
            num_dim (int, optional): Dimensionality of the attention mechanism. Defaults to 128.
            is_weighted (bool, optional): Whether to use weighted feature map fusion. Defaults to False.
        """
        super(WSA, self).__init__()
        self.num_dim = num_dim
        self.embed_dim = args.embed_dim
        self.is_weighted = is_weighted
        self.dropout_r = 0.1

        # Multi-Head Attention and Feed-Forward Network layers
        self.mhatt = MHAtt(args)
        self.ffn = FeedForward(self.embed_dim, self.embed_dim * 2)

        # Dropout layers and Layer Normalization
        self.dropout1 = nn.Dropout(self.dropout_r)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.dropout2 = nn.Dropout(self.dropout_r)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        # Learnable weights for weighted feature map fusion
        if is_weighted:
            self.fmp_weight = nn.Parameter(torch.randn(1, self.num_dim, self.embed_dim))

    def forward(self, x, x_mask=None):
        """
        Forward pass of the Word-Sentence Attention module.

        Args:
            x (torch.Tensor): Input tensor.
            x_mask (torch.Tensor, optional): Masking tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        bs = x.shape[0]

        # Apply multi-head attention and feed-forward network, followed by Layer Normalization and dropout
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        if self.is_weighted:
            # Apply weighted feature map fusion if enabled
            x = (
                self.fmp_weight.expand(bs, x.shape[1], x.shape[2])
                .transpose(1, 2)
                .bmm(x)
            )

        return x


class GA(nn.Module):
    def __init__(self, args):
        """
        Global Attention module.

        Args:
            args: Model configuration arguments.
        """
        super(GA, self).__init__()
        self.h = 2  # Number of attention heads
        self.embed_dim = args.embed_dim
        self.d_k = self.embed_dim // self.h  # Dimensionality of each head

        # Linear transformations for query, key, and gate
        self.linears = clones(nn.Linear(self.embed_dim, self.embed_dim), 3)

        # Linear layers for query, key, and gate
        self.fc_q = nn.Linear(self.d_k, self.d_k)
        self.fc_k = nn.Linear(self.d_k, self.d_k)
        self.fc_g = nn.Linear(self.d_k, self.d_k * 2)

    def forward(self, cap_emb):
        """
        Forward pass of the Global Attention module.

        Args:
            cap_emb (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        bs = cap_emb.shape[0]

        # Linear transformations for query, key, and value
        q, k, v = [
            l(x).view(bs, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (cap_emb, cap_emb, cap_emb))
        ]

        # Gate mechanism
        G = self.fc_q(q) * self.fc_k(k)
        M = torch.sigmoid(self.fc_g(G))  # (bs, h, num_region, d_k*2)
        q = q * M[:, :, :, : self.d_k]
        k = k * M[:, :, :, self.d_k :]

        # Compute attention scores and apply softmax
        scores = torch.div(
            torch.matmul(q, k.transpose(-2, -1)),
            math.sqrt(self.d_k),
            rounding_mode="floor",
        )
        p_attn = torch.softmax(scores, dim=-1)

        # Compute the weighted sum of values
        x = torch.matmul(p_attn, v)
        x = x.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)

        return x


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, use_relu=True):
        """
        Fully connected layer with optional dropout and ReLU activation.

        Args:
            in_size (int): Size of the input features.
            out_size (int): Size of the output features.
            dropout (float, optional): Dropout probability. Default is 0.0.
            use_relu (bool, optional): Whether to apply ReLU activation. Default is True.
        """
        super(FC, self).__init__()
        self.dropout_r = dropout
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the fully connected layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        """
        Feedforward neural network with dropout.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward network.
            dropout (float, optional): Dropout probability. Default is 0.0.
        """
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass of the feedforward neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout=0.0, use_relu=True):
        """
        Multilayer Perceptron (MLP) with optional dropout.

        Args:
            in_size (int): Input size.
            mid_size (int): Hidden layer size.
            out_size (int): Output size.
            dropout (float, optional): Dropout probability. Default is 0.0.
            use_relu (bool, optional): Whether to use ReLU activation. Default is True.
        """
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout=dropout, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.linear(self.fc(x))
        return out


# ====================
# Some Reuse Function
# ====================
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs"""
    im = l2norm(im, dim=-1)
    s = l2norm(s, dim=-1)
    w12 = im.mm(s.t())
    return w12


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def factory(args, cuda=True, data_parallel=False):
    """
    Factory function to create and initialize the model.

    Args:
        args: Namespace containing model configuration and parameters.
        cuda (bool, optional): Flag indicating whether to use CUDA (GPU). Defaults to True.
        data_parallel (bool, optional): Flag indicating whether to use data parallelism. Defaults to False.

    Returns:
        nn.Module: Initialized model instance.
    """
    # Create a copy of args to avoid modifying the original object
    args_new = copy.copy(args)

    # Initialize the model without DistributedDataParallel (DDP)
    model_without_ddp = UrbanCross(
        args_new,
    )

    # Move the model to GPU if cuda is True
    if cuda:
        model_without_ddp.cuda(args_new.gpuid)

    # Apply data parallelism if data_parallel is True
    if data_parallel:
        # Convert BatchNorm layers to SyncBatchNorm for distributed training
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
        # Initialize DistributedDataParallel with model and GPU device ID
        model = DistributedDataParallel(
            model, device_ids=[args.gpuid], find_unused_parameters=False
        )
        # Get the module attribute of the model, as DDP wraps the model with an additional layer
        model_without_ddp = model.module
        # Ensure CUDA is enabled if data parallelism is used
        if not cuda:
            raise ValueError

    return model_without_ddp


def factory_finetune(args, word2idx, cuda=True, data_parallel=False):
    args_new = copy.copy(args)

    # model_without_ddp = SWAN(args_new, word2idx)
    model_without_ddp = UrbanCross_finetune(args_new, word2idx)

    if cuda:
        model_without_ddp.cuda(args_new.gpuid)

    if data_parallel:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
        model = DistributedDataParallel(
            model, device_ids=[args.gpuid], find_unused_parameters=False
        )
        model_without_ddp = model.module
        if not cuda:
            raise ValueError

    return model_without_ddp
