# Flamingo

https://arxiv.org/abs/2204.14198

![image.png](Flamingo/image.png)

1. **Visual Encoding (Frozen Vision Encoder)**
    
    Images or video frames are passed through a *frozen* vision encoder (e.g., EfficientNet / ViT), producing a large set of visual patch tokens.
    
2. **Perceiver Resampler (Trainable)**
    
    **A set of learnable latent queries** attends to the visual tokens via cross-attention.
    
    This produces a *compact, fixed-length sequence* of visual embeddings that are aligned with the language space.
    
3. **Frozen Language Model (LLM Backbone)**
    
    The main LLM (e.g., Chinchilla-style model) is *completely frozen*.
    
    Its original transformer layers (self-attention + MLP blocks) remain unchanged.
    
4. **Gated Cross-Attention Layers (Trainable Add-Ons)**
    
    At certain layers inside the LLM, Flamingo inserts **new trainable gated cross-attention modules**.
    
    During text generation, these modules allow the LLM to query the visual embeddings.
    
5. **Autoregressive Text Generation**
    
    The LLM generates output tokens (answers, captions, descriptions) while having access to the image/video representations at every decoding step through the gated cross-attention layers.
    

### **✅**

### **Trainable Components**

1. **Perceiver Resampler**
    - Learnable latent queries
    - Cross-attention layers
    - Layer norms
        
        → Responsible for compressing and aligning visual features.
        
2. **Gated Cross-Attention Layers inside the LLM**
    - Cross-attention weights
    - Learnable gate scalar(s)
        
        → Enable multimodal fusion without modifying the core LLM weights.
        
3. **Small projection layers (where applicable)**
    - Minor linear projections for modality alignment