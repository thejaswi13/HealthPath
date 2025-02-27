# HealthPath - Your Wellness Companion
HealthPath is a personalized health assistant built for a hackathon, using clustering and a chatbot to deliver tailored wellness insights—designed with women in mind!

## Features
- **Clustering**: Groups you into health levels (0-3) based on BMI, sleep, stress, etc.  
- **Health Insights**: Get custom reports and actionable tips.  
- **Chatbot**: Interactive advice (rule-based in the live demo, LLM-powered locally).  

## Live Demo
Deployed on Streamlit Cloud: [insert your Streamlit URL here]. Uses `app_rule.py` with a rule-based chatbot for cost-free, reliable access—hosting an LLM like Llama 3.2-1B costs money (e.g., $5+/month), so I kept it simple.

## Files
- `app.py`: LLM version with Llama 3.2-1B chatbot (local use).  
- `app_rule.py`: Rule-based version (deployed).  
- `health_data_synthetic.csv`: Required dataset.  

## Run Locally
### LLM Version (`app.py`)
1. Clone: `git clone https://github.com/thejaswi13/HealthPath.git && cd HealthPath`  
2. Install: `pip install streamlit pandas numpy scikit-learn ollama`  
3. Setup Ollama: Install from [ollama.com](https://ollama.com/download), then:  
   - Pull model: `ollama pull llama3.2:1b`  
   - Run server: `ollama serve` (keep this terminal running)  
4. Launch: `streamlit run app.py` (needs `health_data_synthetic.csv`). Visit `http://localhost:8501`.  

### Rule-Based Version (`app_rule.py`)
1. Clone: `git clone https://github.com/thejaswi13/HealthPath.git && cd HealthPath`  
2. Install: `pip install streamlit pandas numpy scikit-learn`  
3. Launch: `streamlit run app_rule.py` (needs `health_data_synthetic.csv`). Visit `http://localhost:8501`.

## Why Rule-Based Live?
The LLM version uses Llama 3.2-1B but requires a local server. I deployed the rule-based version to avoid hosting costs for the hackathon.



## Made With
- Python, Streamlit, Scikit-learn.  
- **LLM**: Llama 3.2-1B for intelligent chatbot insights (local).
