# A comprehensive survey of Vision--Language Models: Pretrained models, fine-tuning, prompt engineering, adapters, and benchmark datasets 

# Abstract:
Vision Language Models (VLMs) have significantly advanced multimodal tasks like image captioning, visual question answering, and multimodal retrieval. This survey presents a systematic review of 115 published papers from 2018 to 2025. It focuses key VLM components including fine-tuning strategies, prompt engineering techniques, pre-trained models, adapter modules, and benchmarking datasets. For each component, we present taxonomies and summarize comparative findings across standard VLM benchmarks. The survey emphasizes the role of lightweight, parameter-efficient adaptation methods in reducing computational overhead while maintaining strong task performance, particularly in real-world deployment contexts. It further examines the strengths and limitations of prompt-based learning, dataset-specific tuning strategies, and architectural trade-offs. Finally, the paper identifies open challenges in scalability, generalization, and bias, and explores emerging research directions including symbolic reasoning, multilingual adaptation, and energy-efficient VLM design. To our best knowledge, this is the first comprehensive survey to integrate these critical components into a single, cohesive survey paper, intended to serve as a foundational resource for researchers and practitioners striving to optimize VLMs for diverse real-world scenarios. The highlights of the review are available at https://github.com/SufyanDanish/VLM-Survey-
# 📌 Journal: Information Fusion
# Paper Link: [Click Here](https://www.sciencedirect.com/science/article/pii/S1566253525006955)

# Vision-Language Models Repository  
This repository provides the latest collection of resources for Vision-Language Models (VLMs). It covers key components essential for research and development in the Vision language model.  

## 📌 Components  

- **Fine-Tuning**:  
- **Pre-trained Models**:   
- **Prompt Engineering**:  
- **Adapters**: 
- **Benchmark Datasets**: 

This repository aims to assist researchers and developers by offering well-organized resources and references to advance work in Vision-Language Modeling.  
if you have the latest paper related to the above-mentioned VLM components then Please share it with us we will include it in the latest resources. 

### Summary of Prompt Engineering Techniques for Vision Language Models (2021–2025)

| Year | Method | Type | Description | Publication | Code |
|------|--------|------|-------------|-------------|------|
| 2021 | CLIP | Hard-Prompt | Introduced contrastive language-image pre-training. | ICML 2021 | [Link](https://github.com/openai/CLIP) |
| 2022 | CoOp | Soft-Prompt | Context Optimization for prompt tuning using learnable embeddings. | IJCV 2022 | [Link](https://github.com/KaiyangZhou/CoOp) |
| 2022 | CPT | Hard-Prompt | Task-specific fine-tuning in vision-language tasks. | - | [Link](https://github.com/thunlp/CPT) |
| 2022 | DenseCLIP | Text Soft-Prompt | Extended CLIP to dense vision tasks with optimized textual prompts. | CVPR 2022 | [Link](https://github.com/raoyongming/DenseCLIP) |
| 2022 | FewVLM | Hard-Prompt | Few-shot learning framework for VLMs using hard prompts. | ACL 2022 | [Link](https://github.com/woojeongjin/FewVLM) |
| 2022 | ProDA | Text Soft-Prompt | Prompt distribution alignment for domain adaptation. | CVPR 2022 | [Link](https://github.com/bbbdylan/proda) |
| 2022 | ProGrad | Text Soft-Prompt | Gradient optimization to improve prompt effectiveness. | CVPR 2022 | [Link](https://github.com/BeierZhu/Prompt-align) |
| 2022 | PEVL | Hard-Prompt | Combined prompt tuning with vision encoders for enhanced alignment. | EMNLP 2022 | [Link](https://github.com/thunlp/PEVL?tab=readme-ov-file) |
| 2022 | VPT | Visual Soft-Prompt | Visual embeddings as learnable prompts. | ECCV 2022 | [Link](https://github.com/kmnp/vpt) |
| 2022 | TPT | Text Soft-Prompt | Enhanced text-based prompt-tuning methods. | NeurIPS 2022 | [Link](https://github.com/azshue/TPT) |
| 2023 | ViPT | Visual Soft-Prompt | Visual Prompt multi-modal Tracking for various downstream tasks. | CVPR 2023 | [Link](https://github.com/jiawen-zhu/ViPT) |
| 2023 | MaPLe | Visual-text & Modal-Prompt | Multi-modal Adaptive Prompt Learning. | CVPR 2023 | [Link](https://github.com/muzairkhattak/multimodal-prompt-learning) |
| 2023 | KgCoOp | Text Soft-Prompt | Knowledge-guided Context Optimization. | CVPR 2023 | [Link](https://github.com/htyao89/KgCoOp) |
| 2023 | LASP | Text Soft-Prompt | Text-to-Text Optimization for Language-Aware Soft Prompting. | CVPR 2023 | [Link](https://www.adrianbulat.com/lasp) |
| 2023 | DAM-VP | Visual Soft-Prompt | Diversity-Aware Meta Visual Prompting. | CVPR 2023 | [Link](https://github.com/shikiw/DAM-VP) |
| 2023 | TaskRes | Text Soft-Prompt | Task Residual for Tuning Vision-Language Models. | CVPR 2023 | [Link](https://github.com/geekyutao/TaskRes) |
| 2023 | RPO | Text Hard-Prompt | Read-only Prompt Optimization for Few-shot Learning. | ICCV 2023 | [Link](https://github.com/mlvlab/rpo) |
| 2023 | PromptSRC | Visual-Text Soft-Prompt | Semantic-Rich Contextual Prompting. | ICCV 2023 | [Link](https://github.com/muzairkhattak/PromptSRC) |
| 2024 | DePT | Visual-Text Soft-Prompt | Dense Prompt Tuning. | CVPR 2024 | [Link](https://github.com/Koorye/DePT) |
| 2024 | TCP | Text Soft-Prompt | Text-Conditioned Prompting. | CVPR 2024 | [Link](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning) |
| 2024 | MMA | Visual-Text Soft-Prompt | Multi-Modal Adaptive Prompting. | CVPR 2024 | [Link](https://github.com/ZjjConan/Multi-Modal-Adapter) |
| 2024 | HPT | Visual-Text Soft-Prompt | Hierarchical Prompt Tuning. | AAAI 2024 | [Link](https://github.com/Vill-Lab/2024-AAAI-HPT) |
| 2024 | CoPrompt | Soft-Prompt | Contextual Prompt Learning. | ICLR 2024 | [Link](https://github.com/ShuvenduRoy/CoPrompt) |
| 2024 | CasPL | Visual-Text Soft-Prompt | Cascade Prompt Learning. | ECCV 2024 | [Link](https://github.com/megvii-research/CasPL) |
| 2024 | PromptKD | Visual-Text Soft-Prompt | Knowledge Distillation-based Prompt Tuning. | CVPR 2024 | [Link](https://github.com/zhengli97/PromptKD) |
| 2025 | DPC | Visual-Text Soft-Prompt | Dual-Prompt Collaboration for tuning VLMs. | CVPR 2025 | [Link](https://github.com/JREion/DPC) |
| 2025 | 2SFS | Visual-Text Soft-Prompt | Two-Stage Few-Shot adaptation for VLMs. | CVPR 2025 | [Link](https://github.com/FarinaMatteo/rethinking_fewshot_vlms) |
| 2025 | MMRL | Visual-Text Soft-Prompt | Multi-Modal Representation Learning | CVPR 2025 | [Link](https://github.com/yunncheng/MMRL) |
| 2025 | NLPrompt | Text Soft-Prompt | Noise-Label Prompt Learning | CVPR 2025 | [Link](https://github.com/qunovo/NLPrompt) |
| 2025 | TAC | Text Soft-Prompt | Task-Aware Clustering for Prompting | CVPR 2025 | [Link](https://github.com/FushengHao/TAC) |
| 2025 | TextRefiner | Text Soft-Prompt | Internal visual features as prompt refiners | AAAI 2025 | [Link](https://github.com/xjjxmu/TextRefiner) |
| 2025 | ProText | Text Soft-Prompt | Prompting with text-only supervision | AAAI 2025 | [Link](https://github.com/muzairkhattak/ProText) |


### Comparative Overview of Vision-Language Models (Pre-2023 vs Post-2023)

| Model | Year | Fine-Tuning Strategy | Architecture | Pretraining Objective | Pretrained Backbone Model | Vision Encoder / Tokenizer | Parameters | Training Data | Key Innovations |
|-------|------|----------------------|--------------|------------------------|----------------------------|-----------------------------|------------|----------------|------------------|
| ViLBERT | 2019 | Full fine-tuning | Two-stream Transformer | Image-text alignment with co-attentional modules | BERT | Object-based features (Faster R-CNN) | 110M | COCO, Conceptual Captions | Co-attentional streams for image-text fusion |
| VisualBERT | 2019 | Full fine-tuning | Single-stream Transformer | Masked language modeling with visual embeddings | BERT | Object-based features (Faster R-CNN) | 110M | COCO, VQA | Shared encoder for text and image inputs |
| CLIP | 2021 | Zero-shot inference | Encoder-decoder | Contrastive image-text learning | Pretrained from scratch | ViT/ResNet | 63M–355M | 400M web image-text pairs | Contrastive learning with dual encoders; broad generalization via natural language supervision |
| ALIGN | 2021 | Zero-shot inference | Dual encoder (EffNet-L2 + Transformer) | Contrastive image-text alignment | EfficientNet-L2 | EfficientNet | 1.8B | 1B+ noisy web image-text pairs | Large-scale noisy training with CLIP-style contrastive objectives |
| SimVLM | 2021 | Full fine-tuning | Unified Transformer encoder-decoder | Prefix language modeling (unified vision-text sequence) | BERT + ResNet | Unified Transformer | 1B+ | Vision-language pairs | Simplified architecture with prefix modeling and no region-level supervision |
| Florence | 2022 | Full fine-tuning | Unified Transformer | Unified encoder with multi-task supervision | Swin Transformer | Swin | 892M | Multilingual web-scale dataset | High-performance universal VL encoder |
| Flamingo | 2022 | Few-shot in-context learning | Perceiver Resampler + Decoder-only Transformer | Frozen vision-language backbones with trainable cross-attention | Chinchilla | ViT-L/14 + Perceiver | 80B | M3W, ALIGN | In-context few-shot learning with frozen backbones and cross-modal fusion |
| BLIP-2 | 2023 | Modular fine-tuning via Q-Former | Vision encoder + frozen LLM + Q-Former | Two-stage: vision-text + vision-to-language generation | ViT-G + OPT/FlanT5 | ViT-G + Q-Former | 223M–400M | WebLI, COCO, CC3M, CC12M | Q-Former for modular downstream tasks |
| IDEFICS | 2023 | Parameter-efficient tuning | Unified Transformer with vision encoder | Instruction-tuned vision-language | OPT + ViT | ViT | 80B | COCO, VQAv2, A-OKVQA | Open-source instruction-following VLM |
| PaliGemma 2 | 2024 | LoRA, fine-grained adapters | Transformer encoder-decoder | Multilingual + synthetic datasets | Gemma + ViT | ViT | - | Synthetic + real data (DOCCI, LAION, CC12M) | Multilingual generation + grounding |
| Gemini 2.0 | 2024 | Modular fine-tuning | PaLM-based encoder-decoder + vision module | Multimodal pretraining with sparse transformers | PaLM 2 + ViT | ViT | - | Multilingual, synthetic corpus | Flexible and efficient multimodal reasoning |
| Kosmos-2.5 | 2024 | Selective fine-tuning | Decoder-only Transformer with ViT + resampler | Document text recognition + image-to-Markdown generation | - | ViT-G/14, ViT-L/14 | 1.3B | Document images, OCR, structured markup data | Layout-aware multimodal literacy via visual-text fusion with Markdown generation |
| GPT-4V | 2024 | No tuning (chat interface) | Unified Transformer with vision-text fusion | Text + image pretraining | GPT-4 | Custom ViT-like encoder | - | Vision-language aligned corpus | GPT-4 vision support with image-text joint encoding |
| Claude 3 Opus | 2024 | Supervised fine-tuning via API | Encoder-decoder transformer | Proprietary encoder-decoder | Proprietary | - | - | Multimodal benchmarks | Safe and high-performance multimodal chat |
| LongVILA | 2024 | Efficient parameter tuning | Video-based encoder-decoder transformer | Video-language transformer | Custom video model | Patch + frame tokenizer | - | Long video, image sequences | Long-context video QA and interleaved image-text reasoning |
| Molmo | 2024 | Instruction tuning | Encoder-decoder transformer | Transformer-based VLM | - | ViT-L/14 (CLIP) | 72B | Open PixMo data | Open-source transparent training |
| Qwen 2.5 VL | 2025 | Instruction tuning | Transformer decoder with visual patch input | Vision transformer + LLM fusion | Qwen 2.5 + ViT | ViT | 3B/7B/72B | Docs, images, audio | OCR + document QA specialization |
| DeepSeek Janus | 2025 | Adapter-based fine-tuning | Dual-stream Transformer with MoE | Multimodal instruction-following | DeepSeek + ViT | ViT | 7B | Instruction + synthetic datasets | Efficient MoE-based dual-stream VLM |
| MiniCPM-o 2.6 | 2025 | Plug-in modules + instruction tuning | Modular lightweight Transformer | Multimodal instruction-following + OCR | MiniCPM + LLaMA3 | Vision adapter | 8B | Instruction-tuned corpus | GPT-4V-level OCR + real-time video understanding on-device |
| Moondream | 2025 | Minimal fine-tuning | Decoder-only Transformer | Multimodal pretraining | - | Compact encoder | 1.86B | Open efficient datasets | Small footprint with privacy focus |
| Pixtral | 2025 | Instruction tuning | Dual-stream compact transformer | Mistral-style ViT + LLM | Mistral + ViT | ViT | 12B | Multi-domain open-source corpus | ViT fusion in compact architecture |




### Dataset Audit Table: Overview of Key Datasets Used in Vision-Language Research

| Dataset | Size | Modalities | Language(s) | Category Diversity | Known Biases / Limitations |
|---------|------|------------|-------------|---------------------|-----------------------------|
| MS COCO | 328K images | Image-Text | English | 91 object categories | Western-centric content; limited cultural diversity; object-centric focus |
| VQAv2 | 204K images; 1.1M Q&A pairs | Image-QA | English | Everyday scenes with varied Q&A | Language bias; answer priors; question redundancy |
| RadGraph | 221K reports; 10.5M annotations | Text (Radiology reports) | English | Radiology findings | Domain-specific; requires medical expertise for annotation; limited to chest X-rays |
| GQA | 113K images; 22M questions | Image-QA | English | Compositional reasoning | Synthetic question generation; potential over-reliance on scene graphs |
| GeoBench-VLM | 10K+ tasks | Satellite-Text | English | Natural disasters, terrain, infrastructure | Sparse labels; coverage gaps |
| SBU Captions | 1M images | Image-Text | English | Web-sourced everyday scenes | Noisy captions; duplicate entries |
| MIMIC-CXR | 377K images; 227K studies | Image-Text | English | Chest X-rays | Hospital-centric; privacy restrictions |
| EXAMS-V | 20,932 questions | Mixed Multimodal | 11 languages | Exam-style reasoning across disciplines | Regional bias; multilingual challenge |
| RS5M | 5M images | Satellite-Text | English | Remote sensing imagery | Sparse labels; class imbalance; varying image quality |
| VLM4Bio | 30K instances | Image-Text-QA | English | Biodiversity, taxonomy | Domain-specific; taxonomic bias; limited generalizability |
| PMC-OA | ~1.65M image-text pairs | Image-Text-QA | English | High diversity within the biomedical domain; Covers a wide range of diagnostic procedures, disease types, and medical findings | Caption noise; requires medical expertise |
| WebLI-100B | 100 Billion image-text pairs | Image-Text | 100+ languages | Global content | Cultural/geographic bias; noisy data |




### Datasets for Vision-Language Models

| Dataset Type | Dataset Name | Description | Applications | Link |
|--------------|--------------|-------------|--------------|------|
| Detection | COCO | 330k images with annotations for detection and segmentation. | Object detection, instance segmentation, image captioning. | [Link](https://cocodataset.org/) |
| Detection | Open Images | 9M+ annotated images for detection. | Object detection, captioning, visual relationship detection. | [Link](https://storage.googleapis.com/openimages/web/index.html) |
| Classification | ImageNet | 14M labeled images across 1K classes. | Image classification, transfer learning. | [Link](http://www.image-net.org/) |
| Classification | Visual Genome | 108k images with scene graphs and object annotations. | VQA, object detection, scene understanding. | [Link](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) |
| Segmentation | ADE20K | 20k images labeled across 150 categories. | Semantic segmentation, scene parsing. | [Link](http://groups.csail.mit.edu/vision/datasets/ADE20K/) |
| Segmentation | Cityscapes | Urban scenes with pixel-level annotations. | Autonomous driving, semantic segmentation. | [Link](https://www.cityscapes-dataset.com/) |
| Text-to-Image | Flickr30k | 31k images with 5 captions each. | Image captioning, text-to-image generation. | [Link](https://www.kaggle.com/datasets/awsaf49/flickr30k-dataset) |
| Text-to-Image | COCO Captions | Subset of COCO with image captions. | Captioning, text-image synthesis. | [Link](https://cocodataset.org/#captions-2015) |
| Multimodal Alignment | VQA | 200k+ questions over 100k images. | Visual QA, multimodal reasoning. | [Link](https://visualqa.org/) |
| Multimodal Alignment | EndoVis-18-VLQA | QA pairs for surgical/endoscopic videos. | Medical QA, surgical assistance. | [Link](https://github.com/longbai1006/Surgical-VQLA?tab=readme-ov-file) |
| Multimodal Alignment | VLM4Bio | 469k QA pairs, 30k images for biodiversity tasks. | Scientific QA, bio-research. | [Link](https://github.com/Imageomics/VLM4Bio) |
| Multimodal Alignment | MS-COCO Text-Only | Captions-only version of MS-COCO. | Text-based retrieval, text-image matching. | [Link](https://www.microsoft.com/en-us/research/project/ms-coco/) |
| Pre-training | Conceptual Captions | 3.3M web-sourced image-caption pairs. | Vision-language pretraining. | [Link](https://github.com/google-research-datasets/conceptual-captions) |
| Pre-training | PathQA Bench Public | 456k pathology QA pairs for PathChat. | Pathology education, clinical AI. | [Link](https://github.com/fedshyvana/pathology_mllm_training?tab=readme-ov-file) |
| Pre-training | SBU Captions | 1M web-collected image-caption pairs. | Captioning, multimodal learning. | [Link](https://www.kaggle.com/datasets/akashnuka/sbucaptions) |
| Multimodal Retrieval | Flickr30k Entities | Object-level annotations on Flickr30k. | Caption alignment, image-text retrieval. | [Link](https://bryanplummer.com/Flickr30kEntities/) |
| Multimodal Retrieval | RS5M | 5M satellite images with English descriptions. | Remote sensing, domain-specific VLM tuning. | [Link](https://github.com/om-ai-lab/rs5m?tab=readme-ov-file) |
| Multimodal Retrieval | v-SRL | Visual elements annotated with semantic roles. | Multimodal grounding, semantic role labeling. | [Link](https://github.com/ahmedssabir/Textual-Visual-Semantic-Dataset-for-Text-Spotting) |
| Multimodal Reasoning | CLEVR | Synthetic QA over generated scenes. | Visual reasoning, compositional QA. | [Link](https://cs.stanford.edu/people/jcjohns/clevr/) |
| Multimodal Reasoning | GMAI-MMBench | 284 datasets across 38 medical modalities. | Medical QA, clinical AI benchmarking. | [Link](https://uni-medical.github.io/GMAI-MMBench.github.io/#2023xtuner) |
| Multimodal Reasoning | NavGPT-Instruct-10k | 10k steps for navigation-based QA. | Navigational reasoning, autonomous systems. | [Link](https://huggingface.co/datasets/ZGZzz/NavGPT-Instruct) |
| Multimodal Reasoning | GQA | 22M compositional QA pairs. | Visual reasoning, scene-based QA. | [Link](https://cs.stanford.edu/people/dorarad/gqa/) |
| Multimodal Reasoning | EXAMS-V | 20,932 multilingual questions in 20 subjects. | Multilingual education QA, model benchmarking. | [Link](https://www.imageclef.org/2025/multimodalreasoning) |
| Multimodal Reasoning | MMVP-VLM | QA pairs for evaluating pattern understanding. | Visual pattern QA, image-text alignment. | [Link](https://tsb0601.github.io/mmvp_blog/) |
| Semantic Segmentation | ADE20K | 20k images across diverse scenes. | Semantic segmentation, scene understanding. | [Link](http://groups.csail.mit.edu/vision/datasets/ADE20K/) |
| Semantic Segmentation | Cityscapes | Urban street views with labels. | Road scene analysis, autonomous driving. | [Link](https://www.cityscapes-dataset.com/) |
| Cross-Modal Transfer | MIMIC-CXR | Chest X-rays + radiology reports. | Clinical VLMs, cross-modal training. | [Link](https://physionet.org/content/mimic-cxr/2.0.0/) |
| Cross-Modal Transfer | MedNLI | Medical Natural Language Inference dataset. | Textual reasoning in medicine. | [Link](https://paperswithcode.com/dataset/mednli) |


### Overview of Vision-Language Datasets for the Medical Domain

| Dataset Name | Image-Text Pairs | QA Pairs | Description | Application | Link |
|--------------|------------------|----------|-------------|-------------|------|
| VQA-Med 2020 | ❌ | ✅ | Dataset for VQA and VQG in radiology; includes questions about abnormalities and image-based question generation. | Medical diagnosis, clinical decision support, multimodal QA | [Link](https://github.com/abachaa/VQA-Med-2020) |
| ROCO | ✅ | ❌ | Multimodal dataset from PubMed Central with radiology/non-radiology images, captions, keywords, and UMLS concepts. | Captioning, classification, retrieval, VQA | [Link](https://github.com/razorx89/roco-dataset?tab=readme-ov-file) |
| VQA-Med 2019 | ❌ | ✅ | 3,200 radiology images with 12,792 QA pairs across 4 categories: Modality, Plane, Organ system, Abnormality. | Medical image analysis, radiology AI, education | [Link](https://github.com/abachaa/VQA-Med-2019) |
| MIMIC-NLE | ✅ | ❌ | 377K chest X-rays with structured labels derived from free-text radiology reports. | Image understanding, NLP for radiology, decision support | [Link](https://github.com/maximek3/MIMIC-NLE) |
| SLAKE | ❌ | ✅ | Bilingual Med-VQA dataset with semantic labels and a medical knowledge base, covering many body parts and modalities. | Annotations, diverse QA, knowledge-based AI | [Link](https://www.med-vqa.com/slake/) |
| GEMeX | ✅ | ✅ | Largest chest X-ray VQA dataset with explainability annotations to enhance visual reasoning in healthcare. | Med-VQA, visual reasoning, explainable AI | [Link](https://www.med-vqa.com/GEMeX/) |
| MS-CXR | ✅ | ❌ | 1,162 image-sentence pairs with bounding boxes and phrases for 8 findings, supporting semantic modeling. | Radiology annotation, contrastive learning, semantic modeling | [Link](https://physionet.org/content/ms-cxr/0.1/) |
| MedICaT | ✅ | ❌ | 217K figures from 131K medical papers with captions, subfigure tags, and inline references. | Captioning, multimodal learning, retrieval | [Link](https://github.com/allenai/medicat) |
| 3D-RAD | ❌ | ✅ | 3D Med-VQA dataset using 4,000+ CT scans and 12,000+ QA pairs, including anomaly and temporal tasks. | 3D VQA, multi-temporal diagnosis, 3D understanding | [Link](https://github.com/Tang-xiaoxiao/M3D-RAD) |
| ImageCLEFmed-MEDVQA-GI | ✅ | ✅ | 10K+ endoscopy images with 30K+ (synthetic) QA pairs, focused on gastrointestinal diagnosis. | GI image analysis, synthetic data, endoscopy VQA | [Link](https://github.com/simula/ImageCLEFmed-MEDVQA-GI-2025) |
| BIOMEDICA | ✅ | ❌ | Over 24M image-text pairs from 6M biomedical articles across various disciplines, for generalist VLMs. | Biomedical VLM pretraining, retrieval, generalist AI | [Link](https://minwoosun.github.io/biomedica-website/) |
| RadGraph | ❌ | ❌ | Annotated chest X-ray reports with clinical entities and relations; structured knowledge from unstructured text. | Info extraction, knowledge graphs, NLP for radiology | [Link](https://aimi.stanford.edu/datasets/radgraph-chexpert-results) |
| PMC-OA | ✅ | ❌ | 1.6M image-caption pairs from PubMed Central OA articles; used in PMC-CLIP training. | Medical retrieval, classification, multimodal learning | [Link](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/PMC-OA.md) |
| ReasonMed | ❌ | ✅ | 370K VQA samples for complex reasoning, generated via multi-agent CoT for explainable answers. | Medical reasoning, clinical QA, explainable AI | [Link](https://huggingface.co/datasets/lingshu-medical-mllm/ReasonMed) |
| Lingshu | ✅ | ✅ | Aggregates 9.3M samples from 60+ datasets for generalist Med-VLMs across QA, reporting, and consultation. | Multimodal QA, report generation, medical dialogue | [Link](https://alibaba-damo-academy.github.io/lingshu/) |
| GMAI-VL-5.5M | ✅ | ✅ | 5.5M medical image-text pairs merged from multiple datasets, for general AI and clinical decision tasks. | General medical AI, QA, diagnosis, multimodal systems | [Link](https://arxiv.org/abs/2411.14522) |



### Table 11: Comparison of Models Across Image Captioning, VQA, and Retrieval Tasks

| Model | Task | Dataset | Metric | Score |
|-------|------|---------|--------|-------|
| Unified VLP [287] | Image Captioning | COCO, Flickr30K | BLEU-4 / CIDEr | 36.5 / 116.9 (COCO), 30.1 / 67.4 (Flickr) |
| VinVL [288] | Image Captioning | COCO | BLEU-4 / CIDEr | 40.9 / 140.9 |
| SimVLM [289] | Image Captioning | COCO | BLEU-4 / CIDEr | 40.3 / 143.3 |
| BLIP [35] | Image Captioning | COCO | BLEU-4 / CIDEr | 41.7 / 143.5 |
| RegionCLIP [290] | Image Captioning | COCO | BLEU-4 / CIDEr | 40.5 / 139.2 |
| BLIP-2 [236] | Image Captioning | COCO, NoCaps | BLEU-4 / CIDEr | 43.7 / 123.7 (COCO), – (NoCaps) |
| FIBER [291] | Image Captioning | COCO | CIDEr | 42.8 |
| NLIP [292] | Image Captioning | Flickr30K | CIDEr | 135.2 |
| LCL [293] | Image Captioning | COCO | CIDEr | 87.5 |
| Unified VLP [287] | VQA | VQA 2.0 | VQA Score | 70.3% |
| VinVL [288] | VQA | VQA 2.0 | VQA Score | 76.6% |
| FewVLM [123] | VQA | VQA 2.0 | VQA Score | 51.1% |
| SimVLM [289] | VQA | VQA 2.0 | VQA Score | 24.1% |
| BLIP [35] | VQA | VQA 2.0 | VQA Score | 77.5% |
| BLIP-2 [236] | VQA | VQA 2.0 | VQA Score | 79.3% |
| VILA [164] | VQA | VQA 2.0, GQA | VQA Score | 80.8% (VQA 2.0), 63.3% (GQA) |
| LCL [293] | VQA | VQA 2.0 | VQA Score | 73.4% |
| TCL [294] | Image Retrieval | COCO, Flickr30K | R@1 | 62.3% / 88.7% |
| CLIP [5] | Image Retrieval | COCO, Flickr30K | R@1 | 58.4% / 88.0% |
| NLIP [292] | Image Retrieval | COCO | R@1 | 82.6% |
| Cross-Attn [295] | Image Retrieval | COCO, Flickr30K | R@1 | 67.8% / 88.9% |
| DreamLIP [296] | Image Retrieval | COCO, Flickr30K | R@1 | 58.3% / 87.2% |

🚀 *Contributions and suggestions are welcome!*  
Contact: sufyandanish@sju.ac.kr
