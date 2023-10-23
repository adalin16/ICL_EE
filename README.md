
# Impact of Sample Selection on In-Context Learning for Entity Extraction from Scientific Writing

This repository provides the system used in our work for in-context learning (ICL) sample selection methods for scientific entity extraction task.

## Installation

Install the required libraries

```
pip install easyinstruct -i https://pypi.org/simple
pip install --upgrade openai
pip install transformers
pip install datasets
```


## Datasets

We use five scientific entity extraction datasets.
- ADE
- MeasEval
- SciERC
- STEM-ECR
- WLPC

## Results

### Main Results

| **Method**                          	| **ADE** 	| **MeasEval** 	| **SciERC** 	| **STEM-ECR** 	| **WLPC** 	|
|-------------------------------------	|---------	|--------------	|------------	|----------	|----------	|
| Baseline Models                     	|         	|              	|            	|          	|          	|
| RoBERTa                             	| **90.42**   	| **56.68**        	| **68.52**      	| **69.70**    	| 28.36    	|
| Zero-shot                           	| 71.29   	| 19.65        	| 17.86      	| 28.89    	| 31.64    	|
| Random                              	| 74.56   	| 22.49        	| 29.27      	| 26.85    	| 32.20    	|
| In-context sample selecting methods 	|         	|              	|            	|          	|          	|
| KATE                                	| 83.11   	| 22.75        	| 29.97      	| 30.78    	| 45.02    	|
| Perplexity                          	| 79.13   	| 21.43        	| 31.31      	| 26.57    	| 30.46    	|
| BM25                                	| 77.28   	| 24.72        	| 35.96      	| 25.61    	| 44.14    	|
| Influence                           	| 86.35   	| 27.13        	| 36.47      	| 27.81    	| **45.41**    	|


### Low-Resource Scenario
| **Method**                          	| **ADE** 	| **MeasEval** 	| **SciERC** 	| **STEM-ECR** 	| **WLPC** 	|
|-------------------------------------	|---------	|--------------	|------------	|----------	|----------	|
| RoBERTa full                            	| 90.42   	| 56.68        	| 68.52      	| 69.70    	| 28.36    	|
| Baseline Models                     	|         	|              	|            	|          	|          	|
| RoBERTa %1                             	| 14.32  	| 19.20       	| 10.16      	| 15.42   	| 10.37    	|
| Zero-shot                         	| 71.29   	| 19.65        	| 17.86      	| **28.89**    	| 31.64    	|
| Random %1                              	| 66.53   	| 21.32        	| 25.31      	| 21.38    	| 28.46   	|
| In-context sample selecting methods 	|         	|              	|            	|          	|          	|
| KATE %1                                	| 69.06   	| **24.48**        	| 26.78      	| 26.49    	| 28.97    	|
| Perplexity                          	| 68.83   	| 22.23       	| 26.42      	| 25.48    	| 26.05    	|
| BM25 %1                                	| 72.66   	| 23.39        	| 31.33      	| 24.24    	| **36.73**    	|
| Influence %1                          	| **73.68**  	| 24.21        	| **32.49**      	| 25.01   	| 34.24    	|

## Running Experiments

### Sample Selection
```
python icl_sample.py \
    --data \
    --metric \
    --embed \
    --model \
    --trained \
    --reversed \
    --train_file \
    --test_file

```

### Evaluation
```
python icl_evaluate.py \
    --data --metric \
    --icl_file_name \
    --model \
    --train_file \ 
    --test_file

```
