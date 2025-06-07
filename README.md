# 📊 Invox System

> Exploring Architectural Variants of Language Agents for Speech-to-Structure Transformation

## 🏗️ System Architecture Overview

Invox is an innovative system that transforms speech input into structured form data using advanced language agents. The system consists of six core components:

### Core Components

| Component | Description |
|-----------|-------------|
| 🎤 **Speech Input** | User records audio responses to form questions |
| 📝 **ASR Service** | Whisper API converts speech to text transcription |
| 🧠 **LLM Agents** | ChatGPT, Claude process transcript and generate form answers |
| ✅ **Verification** | DeepSeek ensures semantic consistency and accuracy |
| 📋 **Form Output** | Structured data populated into target form fields |
| 📊 **Analytics** | Performance metrics: latency, accuracy, cost tracking |

## 🔄 Speech-to-Structure Processing Flow

```
🎙️ Audio Capture → 📊 ASR Transcription → 🧠 LLM Processing → ✅ Verification → 📋 Form Population
```

### Processing Steps

1. **🎙️ Audio Capture**: Employee records speech for each form question
2. **📊 ASR Transcription**: Whisper converts audio to text transcript
3. **🧠 LLM Processing**: Language agents analyze transcript with form context
4. **✅ Verification**: Semantic similarity check and consensus logic
5. **📋 Form Population**: Structured data fills target form fields

## 🏛️ Five Architectural Variants

The system implements five different architectural approaches to optimize for various use cases:

### 1. Single-Pass Full Input
- ✅ One LLM processes entire transcript
- ✅ All form questions answered simultaneously
- ✅ Minimal latency, simple orchestration
- ⚠️ May struggle with complex/long forms

### 2. Iterative Single-Question
- ✅ One question processed at a time
- ✅ Focused context per field
- ✅ Higher accuracy for individual fields
- ⚠️ Increased API calls and latency

### 3. Multi-LLM Consensus (Full)
- ✅ Multiple LLMs process full input
- ✅ Consensus via semantic similarity
- ✅ Reduced model-specific bias
- ✅ Enhanced robustness and accuracy

### 4. Multi-LLM Consensus (Iterative)
- ✅ Multiple LLMs per individual question
- ✅ Fine-grained consensus mechanism
- ✅ Maximum accuracy potential
- ⚠️ Highest computational cost

### 5. Hybrid Refinement
- ✅ Initial full-pass + selective refinement
- ✅ Iterative re-query for low confidence
- ✅ Balanced latency and accuracy
- ✅ Adaptive resource allocation

## 👥 User Journey & Roles

### 🔧 Admin Journey
1. Register organization and manage employees
2. Create dynamic form templates with custom fields
3. Configure AI architecture variant per form
4. Assign forms to specific employees
5. Monitor form completion and review results
6. Analyze performance metrics and costs

### 👤 Employee Journey
1. Login to view assigned forms dashboard
2. Select form and view questions one by one
3. Record audio response for each question
4. Review auto-filled form responses
5. Edit or confirm answers as needed
6. Submit completed form for processing

## 💻 Technology Stack

### Frontend
- React.js + TypeScript
- Tailwind CSS
- DaisyUI Components
- Audio Recording APIs

### Backend
- Node.js + Express
- TypeScript
- JWT Authentication
- JSON-RPC Communication

### AI Services
- OpenAI Whisper (ASR)
- ChatGPT API
- Anthropic Claude
- DeepSeek (Verification)

### Infrastructure
- PostgreSQL Database
- Docker Containers
- Redis (Optional)
- Cloud Deployment

## 📈 Evaluation Metrics

The system is evaluated across five key dimensions:

| Metric | Description |
|--------|-------------|
| ⚡ **Latency** | End-to-end processing time from speech input to form completion |
| 🎯 **Accuracy** | Exact match and F1 scores against gold-standard annotations |
| 🔗 **Consistency** | Logical coherence between related form field answers |
| 🔧 **Modularity** | Ease of component replacement and system maintenance |
| 💰 **Cost** | Computational resources and API token usage expenses |

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- PostgreSQL
- Docker (optional)
- API keys for OpenAI, Anthropic, and DeepSeek

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Built with ❤️ for efficient speech-to-structure transformation