CONDA_ENV := lilac
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate $(CONDA_ENV)

.PHONY: all encode retrieve

all: run_visrag

setup:
	@echo "=== [1/2] Setup ==="
	git clone https://github.com/OpenBMB/VisRAG.git
	conda create --name VisRAG python==3.10.8 -y
	conda activate VisRAG
	conda install nvidia/label/cuda-11.8.0::cuda-toolkit
	cd VisRAG
	pip install -r requirements.txt
	pip install -e .
	cd timm_modified
	pip install -e .
	cd ..
	hf download openbmb/VisRAG-Ret --local-dir ./VisRAG-Ret
	export PYTHONPATH="${PYTHONPATH}:$(pwd)"

encode_retrieve: setup
	@echo "=== [2/2] Encoding and Retrieving ==="
	@bash visrag_scripts/eval_retriever/eval.sh 512 2048 4 4 wmean causal InfoVQA VisRAG-Ret

run_visrag: encode_retrieve
	@echo "✅ ALL STEPS COMPLETED SUCCESSFULLY!"