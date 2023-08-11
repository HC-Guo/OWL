# single
## alpaca
![alpaca](./alpaca_bench/model_score/gpt-3.5-turbo_single_radar.png)
## ops
![ops](./ops_bench/model_score/gpt-3.5-turbo_single_radar.png)
## mt
![ops](./mt_bench/model_score/gpt-3.5-turbo_single_radar.png)


# pairwise
## alpaca
![alpaca](./alpaca_bench/model_score/gpt-3.5-turbo_pairwise_each_heatmap.png)
![alpaca](./alpaca_bench/model_score/gpt-3.5-turbo_pairwise_each_stacked_bar.png)
## ops
![ops](./ops_bench/model_score/gpt-3.5-turbo_pairwise_each_heatmap.png)
![ops](./ops_bench/model_score/gpt-3.5-turbo_pairwise_each_stacked_bar.png)
## mt
![ops](./mt_bench/model_score/gpt-3.5-turbo_pairwise_each_heatmap.png)
![ops](./mt_bench/model_score/gpt-3.5-turbo_pairwise_each_stacked_bar.png)


# gpt-3.5-turbo vs gpt-4
测试集：mt_banch中的前3个问题\
（因gpt api调用限制，先粗略评测下; api限制 10,000 tokens per minute (TPM) or 200 requests per minute (RPM)）
## gpt-3.5-turbo
![mt_bench](./mt_bench/model_score/gpt-3.5-turbo_pairwise_each_tmp.png)
## gpt-4
![mt_bench](./mt_bench/model_score/gpt-4_pairwise_each_tmp.png)
