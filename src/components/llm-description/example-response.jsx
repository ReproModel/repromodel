import dedent from "dedent"

const exampleLatexOutput = dedent`
  \section{Method}
  \label{sec:method}

  \begin{figure*}[t]
  \vskip 0.2in
      \centering
      \includegraphics[width=0.7\linewidth]{pic/model.pdf}
      \caption{\textbf{The model achitecture of \onepeace}. The model is composed of multiple components that are specific to each modality (e.g., vision adapter) as well as components that are shared across modalities (e.g., self-attention layer). Different combinations of modules can be used to perform various tasks. For instance, the combination of vision adapter, self-attention layers, and vision FFNs can be utilized for vision tasks.}
      \label{fig:model}
  \end{figure*}

  \subsection{Architecture}
  In this section, we introduce the model architecture of \onepeace, which consist of multiple modality-specific adapters and a modality fusion encoder.
  The overall architecture is shown in Figure~\ref{fig:model}.

  \subsubsection{Modality-Specific Adapters}

  Different modalities exhibit distinct characteristics.
  To facilitate the processing of modality-specific inputs, we design modality-specific adapters to process data from each modality individually.
  Note that these adapters do not interfere with each other and are solely responsible for converting the raw input into vectors.
  This affords us the flexibility to choose the appropriate networks for the adapters, which can be Transformers~\cite{transformer,vit,swin}, CNNs~\cite{cnn,resnet}, RNNs~\cite{rnn,gru,lstm}, etc.
  In this paper, we design three modality-specific adapters for \onepeace, including vision adapter, audio adapter, and language adapter.

  \paragraph{Vision Adapter}
  Given an image $I$, we use a hierarchical MLP (hMLP) stem~\cite{Touvron2022ThreeTE} to patchify the image by gradually increasing the patch size to $16 \times 16$. Note that there is no interaction between different patches. Then the image patches are flattened to a sequence and prepended with a vision class embedding. By adding the absolute positional embeddings into the image embeddings, we derive the image input representation $E^V=\langle \bm{e}^V_{cls}, \bm{e}^V_{1}, \bm{e}^V_{2}, ..., \bm{e}^V_{M} \rangle$, where $M$ denotes the total number of image patches.

  \paragraph{Audio Adapter}
  Given an audio $A$, we set the sample rate to 16kHz and normalized the raw waveform input to zero mean and unit variance. Then the normalized waveform is processed by a convolutional feature extractor~\cite{wav2vec2} to get the audio embeddings. Instead of using the absolute positional embeddings, we use a convolution layer to extract relative position information and add it to the audio embeddings~\cite{Mohamed2019TransformersWC}. With a prepended audio class embedding, we finally obtain the audio input representation $E^A=\langle \bm{e}^A_{cls}, \bm{e}^A_{1}, \bm{e}^A_{2}, ..., \bm{e}^A_{N} \rangle$, where $N$ denotes the length of the audio.

  \paragraph{Language Adapter}
  Given a text $T$, we first apply byte-pair encoding (BPE) \cite{bpe} to transform it to a subword sequence. Two special tokens $\left[{\rm CLS}\right]$ and $\left[{\rm EOS}\right]$ are inserted at the beginning and end of the sentence to indicate its start and end. Then an embedding layer is used to embed the subword sequence to the text embeddings. After summing the text embeddings and absolute positional embeddings, we obtain the text input representation $E^L=\langle \bm{e}^L_{cls}, \bm{e}^L_{1}, \bm{e}^L_{2}, ..., \bm{e}^L_{K}, \bm{e}^L_{eos} \rangle$, where $K$ denotes the text sequence length.

  \subsubsection{Modality Fusion Encoder}
  Following previous works~\cite{simvlm,ofa,coca,beit3}, the modality fusion encoder is based on the Transformer architecture~\cite{transformer}. 
  To make it more suitable for different modalities, we make the following improvements:

  \paragraph{Modality-Specific FFNs} 
  To facilitate the interaction between different modalities while capturing modality-specific information, we set up a shared self-attention layer and three modality-specific feed-forward networks (FFNs) in each Transformer block.
  The shared self-attention layer enables the interaction between different modalities through the attention mechanism.
  The three modality-specific FFNs include vision FFN (V-FFN), audio FFN (A-FFN), and language FFN (L-FFN), which can further extract information within their respective modalities after modality interaction.

  We find it can achieve better results compared with the vanilla Transformer architecture. 
  Furthermore, this structure can be disassembled into different branches to handle the tasks of various modalities.
  For example, the vision adapter, self-attention layer, and vision FFNs can be combined into the vision branch (V-Branch) to process visual tasks.
  Similarly, we named other branches as A-Branch, L-Branch, VA Branch, VL-Branch, AL-Branch, and VAL-Branch.

  \paragraph{Sub-LayerNorm} 
  We introduce Sub-LayerNorm~\cite{magneto} to each Transformer block to increase training stability. 
  Specifically, We insert layer normalization before the input projection and output projection of each self-attention layer and FFN layer. 
  In our preliminary experiments, we find that Sub-LayerNorm can achieve better performance compared to the Pre-LayerNorm~\cite{gpt3}.

  \paragraph{GeGLU Activation Function} 
  To further improve performance, we replace the activation function in FFN with GeGLU~\cite{glu} activation function. 
  The intermediate dimension of FFN is set to $4$ times of the embedding dimension, which is consistent with the practice of PaLM~\cite{palm}.

  \paragraph{Relative Position Bias (RPB)}
  For positional information, we introduce 1D relative position bias~\cite{T5} for text and audio, and 2D relative position bias for image~\cite{coatnet}. 
  At the pre-training stage, the relative position bias of different self-attention layers is shared. 
  At the fine-tuning stage, we decouple the relative position bias of each self-attention layer and let them inherit the weights of the pretrained relative bias.

  \paragraph{LayerScale} 
  We introduce LayerScale~\cite{cait} to dynamically adjust the value of each residual block. Specifically, we multiply the output of each layer (e.g., self-attention layer and FFN) by a learnable diagonal matrix, whose values will be initialized to $1e-6$. In our preliminary experiments, LayerScale is beneficial for stabilizing training and improving performance.

  \begin{figure*}[t]
  \vskip 0.2in
      \centering
      \includegraphics[width=1\linewidth]{pic/pretraining_tasks.pdf}
      \caption{\textbf{The pretraining tasks of \onepeace.} Mask contrastive learning encourages the masked features close to the positive features (indicated by the green lines) and get away from the negative features (indicated by the red lines). Note that we compute the cross-modal contrastive loss by gathering features from all GPU devices, while the mask contrastive loss is computed on the local batch.}
      \label{fig:loss}
  \end{figure*}

  \subsection{Pretraining Tasks}
  In this section, we present the pretraining tasks employed in \onepeace, including cross-modal contrastive learning and mask intra-modal contrastive learning. 
  Cross-modal contrastive learning endows the model with cross-modal retrieval capability, while mask intra-modal contrastive learning enables the model to achieve superior fine-tuning performance on downstream tasks. 
  An illustration of the pretraining tasks is shown in Figure~\ref{fig:loss}.

  \subsubsection{Cross-Modal Contrastive Learning}
  We perform cross-modal contrastive learning to align the semantic space of vision, language, and audio.
  It contains both vision-language contrastive learning and audio-language contrastive learning.
  Given an image-text pair $\left(I, T\right)$, we use the V-Branch and L-Branch to extract the image features and text features, respectively.
  We regard the final output of vision class token and language class token as the global representations.
  Followed by a linear projection and normalization, we obtain the image embedding $\bm{s}^V$ and language embedding $\bm{s}^L$.
  Similarly, given an audio-text pair $\left( A, T"\right)$, we use the A-Branch and L-Branch to obtain the audio embedding $\bm{s}^A$ and language embedding $\bm{s}^{L"}$.
  The contrastive loss functions are shown below:

  \begin{equation}
  \mathcal{L}_{CL-VL} = -\frac{1}{2N}\sum_{i=1}^{N}({\rm log}\frac{{\rm exp}(\bm{s}_i^{V}\bm{s}_i^{L}/\sigma)}{\sum_{j=1}^{N}{\rm exp}(\bm{s}_i^V\bm{s}_j^L/\sigma)} + {\rm log}\frac{{\rm exp}(\bm{s}_i^{V}\bm{s}_i^{L}/\sigma)}{\sum_{j=1}^{N}{\rm exp}(\bm{s}_j^V\bm{s}_i^L/\sigma)}),
  \end{equation}

  \begin{equation}
  \mathcal{L}_{CL-AL} = -\frac{1}{2N}\sum_{i=1}^{N}({\rm log}\frac{{\rm exp}(\bm{s}_i^A\bm{s}_i^{L"}/\sigma)}{\sum_{j=1}^{N}{\rm exp}(\bm{s}_i^A\bm{s}_j^{L"}/\sigma)} + {\rm log}\frac{{\rm exp}(\bm{s}_i^A\bm{s}_i^{L"}/\sigma)}{\sum_{j=1}^{N}{\rm exp}(\bm{s}_j^A\bm{s}_i^{L"}/\sigma)}),
  \end{equation}

  where $\mathcal{L}_{VL}$ and $\mathcal{L}_{AL}$ refer to the vision-language contrastive loss and audio-language contrastive loss, respectively. $N$ is the batch size, $i,j$ are indexes within the batch, and $\sigma$ is a learnable temperature parameter (initialized to $0.07$).

  \subsubsection{Mask Intra-Modal Contrastive Learning}

  Cross-modal contrastive learning mainly focuses on aligning features between different modalities. 
  However, it lacks emphasis on the learning of fine-grained details within modalities, leading to suboptimal finetuning performance in  downstream tasks~\cite{fd-clip}. 
  To address this issue, we introduce mask intra-modal contrastive learning~\footnote{Mask intra-modal contrastive learning is similar to \cite{conmim}, but we extend it to more modalities.}.
  It"s still a form of contrastive learning, but unlike cross-modal contrastive learning, mask intra-modal contrastive learning focuses more on learning the fine-grained details within the modality.
  Our experiment results show that it complements cross-modal contrastive learning and enhances the model"s fine-tuning performance in downstream tasks. 
  Next, we introduce how to apply mask intra-modal contrastive learning to various modalities.

  \paragraph{Mask Vision Contrastive Learning}
  Given an image, we randomly mask $75\%$ patches. Following~\cite{mae}, we only input the unmasked patches to the V-Branch to reduce computation costs and save memory. 
  Then the encoded unmasked patches are concatenated with the mask token and fed to a lightweight Transformer decoder, which generates the corrupted features. 
  In addition, the original image is fed to the vision branch to get the target features.
  The corrupted features are learning to match the target features, the loss function is shown below:

  \begin{equation}
  \label{eq:mcl_v}
  \mathcal{L}_{MCL-V} = -\frac{1}{N\hat{N}_V}\sum_{i=1}^{N}\sum_{j=1}^{\hat{N}_V}{\rm log}\frac{{\rm exp}(\bm{\hat{h}}_{ij}^V \cdot {\rm sg}(\bm{h}_{ij}^V)/\tau)}{\sum_{m=1}^{N}\sum_{n=1}^{N_V}{\rm exp}(\bm{\hat{h}}_{ij}^V  \cdot {\rm sg}(\bm{h}_{mn}^V)/\tau)},
  \end{equation}

  Where $\bm{\hat{h}}_{ij}^V$ denotes the representation of the masked patch, $\bm{h}_{ij}^V$ denotes the representation of the target patch, $\text{sg}(\cdot)$ denotes the stop gradient operation. $N$ is the batch size, $\hat{N}_V$ is the number of masked patches within a sample, $N_V$ is the number of whole patches within a sample. $\tau$ is a constant temperature value, we set it to $0.4$. 
  This loss function not only encourages the masked patches close to the positive patches but also gets away from the negative patches.
  As a result, each patch acquires a unique semantic meaning, which makes the model better transfer to downstream tasks.~\cite{fd-clip}

  \paragraph{Mask Audio Contrastive Learning}
  Given an audio, we sample $p=0.11$ of all time-steps to be starting indices and mask the subsequent $5$ time-steps. We manually adjust the mask rate to make approximately $55\%$ of all time steps to be masked for each sample. 
  Similar to the image, the corrupted audio is encoded to the corrupted features, and the original audio is encoded to the target features.
  The loss function is similar to Eq. (\ref{eq:mcl_v}), denoted as $\mathcal{L}_{MCL-A}$.

  \paragraph{Mask Language Contrastive Learning}
  Given a text, we randomly mask $15\%$ tokens of a text sequence. 
  Different from BERT~\cite{bert}, we don"t replace the masked token with the original tokens or random tokens. Instead, we only feed the unmasked tokens to the L-Branch, and get the final corrupted features through the Transformer decoder. 
  The loss function is denoted as $\mathcal{L}_{MCL-L}$.

  \paragraph{Mask Vision-Language Contrastive Learning}
  Given an image-text pair, We randomly mask $68.75\%$ patches of the image and $40\%$ tokens of the text. 
  We concatenate the unmasked patches and unmasked tokens, and encode them into the corrupted vision-language features through the VL-Branch and the Transformer decoders.
  We also concatenate the original image patches and tokens, and encode them into the target vision-language features.
  We use Equation~\ref{eq:mcl_v} to calculate the contrastive loss between the masked patch and the target patch, and Equation~\ref{eq:mcl_l} to calculate the contrastive loss between the masked token and the target token.
  The average of these two losses forms the mask vision-language contrastive loss, denoted as $\mathcal{L}_{MCL-VL}$.

  \paragraph{Mask Audio-Language Contrastive Learning}
  Given an audio-text pair, we randomly mask $45\%$ time-steps of the audio waveform and $40\%$ tokens of the text.
  After going through a process similar to mask vision-language contrastive learning, we obtain corresponding corrupted features and target features. 
  We denote the loss function of mask audio-language contrastive learning as $\mathcal{L}_{MCL-AL}$.

  \subsection{Two-stage Pretraining}
  \label{sec:two_stage_pretraining}
  We divide the pretraining of \onepeace into two stages: vision-language pretraining and audio-language pretraining.
  During the vision-language pretraining stage, the model only trains on image-text pairs and updates parameters that are relevant to vision and language.
  The loss function at this stage is shown below:

  \begin{equation}
  \mathcal{L}_{VL} = \mathcal{L}_{CL-VL} + 1.0*\mathcal{L}_{MCL-V} + 0.5*\mathcal{L}_{MCL-L} + 1.0*\mathcal{L}_{MCL-VL}
  \end{equation}

  During the audio-language pretraining stage, the model trains solely on audio-text pairs, and we only update A-Adapter, A-FFNs, and other audio-related parameters. 
  The remaining parameters including self-attention layers are totally frozen.
  Despite not training on image-audio pairs, the semantic space between vision and audio is still aligned by using language as the anchor.
  The loss function at the audio-language pretraining stage is shown below:

  \begin{equation}
  \mathcal{L}_{AL} = \mathcal{L}_{CL-AL} + 1.0*\mathcal{L}_{MCL-A} + 1.0*\mathcal{L}_{MCL-AL}
  \end{equation}

  \subsection{Pretraining Details}

  \paragraph{Pretraining Datasets}
  Our pretraining datasets are divided into two parts: image-text pairs and audio-text pairs. 
  For image-text pairs, we use LAION-2B, a dataset obtained by web crawling.
  For audio-text pairs, we collect a large amount of open-source environmental sound datasets.
  For replication, all pretraining datasets are publicly available.
  We provide more details about the pretraining datasets in Appendix~\ref{app:audio_text_data_details}.

  \paragraph{Pretraining Settings}
  \input{table/model_size}
  \input{table/imagenet_result}
  \onepeace is a giant-size model with $4$B parameters. We list the detailed hyper-parameters of \onepeace in Table~\ref{tb:model_configuration}. 
  In addition, we introduce a lightweight Transformer decoder to recover the masked units from the visible units. The decoder is similar to the modality-fusion encoder, each block of it also consists of a shared self-attention layer and three modality-specific FFNs.
  It has $2$ layers with $768$ hidden size, $2048$ intermediate size, and 12 attention heads.
  The model weights of \onepeace are randomly initialized at the beginning, except for the feature extractor of A-adapter, for which we use the weights of WavLM~\cite{wavlm} for initialization. We find that a randomly initialized feature extractor can lead to overfitting, whereas incorporating the weights from WavLM significantly mitigated this problem and improve the performance.
  More details about the pretraining settings are provided in Appendix~\ref{app:pretraining_hyperparameters}.

  \paragraph{Training Acceleration}
  We introduce several model acceleration and memory optimization techniques to train \onepeace. 
  First, we use memory-efficient attention~\cite{memory_efficient_attn,flash_attn} implemented in the Xformers library~\footnote{\url{https://github.com/facebookresearch/xformers/tree/main/xformers}} to save memory usage and improve training speed.
  Secondly, we use the gradient checkpointing~\cite{checkpointing} technique to save memory, which allows us to use a larger batch size to train the model. 
  In addition, we replace the layer normalization with Fused LayerNorm implemented in the Flash Attention library~\footnote{\url{https://github.com/HazyResearch/flash-attention}} and use nvFuser~\footnote{\url{https://github.com/pytorch/tutorials/blob/main/intermediate_source/nvfuser_intro_tutorial.py}} to fuse the operations of dropout, LayerScale, stochastic depth, and residual summing, which can bring some additional speed improvements. 
  To improve the stability of model training and avoid gradient overflow issues, we use Bfloat16 precision to train \onepeace.
`

const examplePlainTextOutput = dedent`
  3. Method
  
    3.1 Architecture
    
    The model architecture of ONE-PEACE consists of three modality adapters and a modality fusion encoder. The overall architecture is shown in Figure 1.
    
    Modality Adapters. We design modality adapters to convert different raw signals into unified features. Note that these adapters do not interact with each other, which affords us the flexibility to choose appropriate networks for them, such as Transformers [26, 27, 84], CNNs [85, 86], RNNs [87, 88], etc. We design three lightweight modality adapters for ONE-PEACE:
    • Vision Adapter (V-Adapter). Given an image, we use a hierarchical MLP (hMLP) stem [89] to patchify the image by gradually increasing the patch size to 16 × 16. There is no interaction between different patches. Then the image patches are flattened into a sequence and prepended with a vision class embedding. By adding the absolute positional embeddings into the image embeddings, the image representation is EV = ⟨eVcls, eV1 , eV2 , ..., eVM ⟩, where M denotes the total number of image patches.
    • Audio Adapter (A-Adapter). Given an audio, we set the sample rate to 16kHz and normalize the raw audio waveform to zero mean and unit variance. Then the normalized waveform is processed by a convolutional feature extractor [11] to get the audio embeddings. Instead of using the absolute positional embeddings, we use a convolution layer to extract relative position information and add it to the audio embeddings [90]. With a prepended audio class embedding, we obtain the audio representation EA = ⟨eAcls, eA1 , eA2 , ..., eAN ⟩, where N denotes the length of the audio representation.
    • Language Adapter (L-Adapter). Given a text, we first apply byte-pair encoding (BPE) [91] to transform it to a subword sequence. Two special tokens [CLS] and [EOS] are inserted at the beginning and end of the sentence to indicate its start and end. Then an embedding layer is used to embed the subword sequence to the text embeddings. After summing the text embeddings with absolute positional embeddings, we obtain the text representation EL = ⟨eLcls, eL1 , eL2 , ..., eLK , eLeos⟩, where K denotes the text sequence length.

    Modality Fusion Encoder. Following previous works [63, 31, 33, 34, 92], the modality fusion encoder is based on the Transformer architecture [26]. We set up a shared self-attention layer and three modality feed-forward networks (FFNs) in each Transformer block. The shared self-attention layer enables the interaction between different modalities through the attention mechanism. The three modality FFNs (V-FFN, A-FFN, and L-FFN) can further extract information within their respective modalities. To stabilize training and enhance model performance, we make the following improvements:
    • Sub-LayerNorm. We incorporate Sub-LayerNorm [93] into each Transformer block to enhance training stability. Specifically, We insert layer normalization before the input projection and output projection of each self-attention layer and FFN layer. In our preliminary experiments, we find that Sub-LayerNorm can achieve better performance compared to the Pre-LayerNorm [94].
    • GeGLU Activation Function. To further improve performance, we replace the activation function in FFN with GeGLU [95] activation function. The intermediate dimension of FFN is set to 4 times the embedding dimension, which is consistent with the practice of PaLM [96].
    • Relative Position Bias (RPB). For positional information, we introduce 1D relative position bias [97] for text and audio, and 2D relative position bias for image [98]. At the pretraining stage, the relative position bias of different self-attention layers is shared. At the fine-tuning stage, we decouple the relative position bias of each self-attention layer and let them inherit the weights of the pretrained relative bias.
    • LayerScale. We use LayerScale [99] to dynamically adjust the output of each residual block. Specifically, before adding to the residual, we multiply the output of each layer (e.g., self-attention layer and FFN) by a learnable diagonal matrix, whose values will be initialized to 1e − 6. In our preliminary experiments, LayerScale is beneficial for stabilizing training and improving performance.

    This "sharing-separated" architecture enables ONE-PEACE to disassemble into different branches that handle tasks for various modalities. For example, the vision adapter, self-attention layer, and vision FFNs can be combined into the vision branch (V-Branch) to process vision tasks. Similarly, we named other branches as audio branch (A-Branch), language branch (L-Branch), vision-audio branch (VA-Branch), vision-language branch (VL-Branch), audio-language branch (AL-Branch), and vision-audio-language branch (VAL-Branch).

    3.2 Tasks

    The pretraining tasks of ONE-PEACE include cross-modal contrastive learning and intra-modal denoising contrastive learning. Cross-modal contrastive learning endows the model with cross-modal retrieval capability, while intra-modal denoising contrastive learning enables the model to achieve superior fine-tuning performance in downstream tasks. An illustration of the pretraining tasks is shown in Figure 2.
    Cross-Modal Contrastive Learning. Cross-modal contrastive learning is a widely-used pretraining task that effec- tively aligns the semantic spaces of different modalities. The key idea of this method is to maximize the similarity of related sample pairs across different modalities while minimizing the similarity of unrelated sample pairs. Given a sample pair (S1, S2) of arbitrary modalities (e.g., image-text pair or audio-text pair), we extract their features using the corresponding branches of ONE-PEACE. The outputs of the special tokens (e.g., vision class token or language class token) are regarded as global representations. Followed by a linear projection and normalization, we obtain the final representations s1 and s2. The loss function is shown below:
`

export { exampleLatexOutput, examplePlainTextOutput }