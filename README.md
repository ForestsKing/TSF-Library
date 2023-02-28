# Long-term Time Series Forecasting Library (LTSFlib)

LTSFlib is an open-source library for long-term time series forecasting task.

To reflect the performance visually and realistically, we test each algorithm fairly with uniform data input and hardware platform. We predict the future 96 timestamps based on the past 96 timestamps on four ETT datasets. We run each subtask for 3 iters and take the average value as the final result.

The detailed results are as follows:

<table>
    <tr>
        <th rowspan="2" style="text-align:center"> Model </th>
        <th colspan="2" style="text-align:center"> ETTh1 </th>
        <th colspan="2" style="text-align:center"> ETTh2 </th>
        <th colspan="2" style="text-align:center"> ETTm1 </th>
        <th colspan="2" style="text-align:center"> ETTm2 </th>
        <th rowspan="2" style="text-align:center"> Paper </th>
        <th rowspan="2" style="text-align:center"> Year </th>
    </tr>
    <tr>
        <th colspan="1" style="text-align:center">MSE</th>
        <th colspan="1" style="text-align:center">MAE</th>
        <th colspan="1" style="text-align:center">MSE</th>
        <th colspan="1" style="text-align:center">MAE</th>
        <th colspan="1" style="text-align:center">MSE</th>
        <th colspan="1" style="text-align:center">MAE</th>
        <th colspan="1" style="text-align:center">MSE</th>
        <th colspan="1" style="text-align:center">MAE</th>
    </tr>
    <!-- 模版
    <tr>
        <td style="text-align:center"> Model </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> 0.0000 </td>
        <td style="text-align:center"> <a href="">Link</a> </td>
        <td style="text-align:center"> 2023 </td> 	
    </tr>
    -->
    <tr>
        <td style="text-align:center"> Reformer </td>
        <td style="text-align:center"> 0.8234 </td>
        <td style="text-align:center"> 0.6690 </td>
        <td style="text-align:center"> 1.6758 </td>
        <td style="text-align:center"> 1.0473 </td>
        <td style="text-align:center"> 0.8355 </td>
        <td style="text-align:center"> 0.6485 </td>
        <td style="text-align:center"> 0.7868 </td>
        <td style="text-align:center"> 0.6837 </td>
        <td style="text-align:center"> <a href="https://arxiv.org/abs/2001.04451">Link</a> </td>
        <td style="text-align:center"> 2020 </td> 	
    </tr>
    <tr>
        <td style="text-align:center"> Informer </td>
        <td style="text-align:center"> 0.9678 </td>
        <td style="text-align:center"> 0.7704 </td>
        <td style="text-align:center"> 3.1124 </td>
        <td style="text-align:center"> 1.3865 </td>
        <td style="text-align:center"> 0.5769 </td>
        <td style="text-align:center"> 0.5348 </td>
        <td style="text-align:center"> 0.4520 </td>
        <td style="text-align:center"> 0.5308 </td>
        <td style="text-align:center"> <a href="https://ojs.aaai.org/index.php/AAAI/article/view/17325">Link</a> </td>
        <td style="text-align:center"> 2021 </td> 	
    </tr>
    <tr>
        <td style="text-align:center"> Autoformer </td>
        <td style="text-align:center"> 0.4578 </td>
        <td style="text-align:center"> 0.4566 </td>
        <td style="text-align:center"> 0.3998 </td>
        <td style="text-align:center"> 0.4286 </td>
        <td style="text-align:center"> 0.5128 </td>
        <td style="text-align:center"> 0.4780 </td>
        <td style="text-align:center"> 0.3109 </td>
        <td style="text-align:center"> 0.3515 </td>
        <td style="text-align:center"> <a href="https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html">Link</a> </td>
        <td style="text-align:center"> 2021 </td> 	
    </tr>
    <tr>
        <td style="text-align:center"> FEDformer </td>
        <td style="text-align:center"> 0.3887 </td>
        <td style="text-align:center"> 0.4248 </td>
        <td style="text-align:center"> 0.3525 </td>
        <td style="text-align:center"> 0.3968 </td>
        <td style="text-align:center"> 0.3977 </td>
        <td style="text-align:center"> 0.4278 </td>
        <td style="text-align:center"> 0.2082 </td>
        <td style="text-align:center"> 0.2932 </td>
        <td style="text-align:center"> <a href="https://proceedings.mlr.press/v162/zhou22g.html">Link</a> </td>
        <td style="text-align:center"> 2022 </td> 	
    </tr>
    <tr>
        <td style="text-align:center"> DLinear </td>
        <td style="text-align:center"> <b>0.3863</b> </td>
        <td style="text-align:center"> 0.4007 </td>
        <td style="text-align:center"> 0.3140 </td>
        <td style="text-align:center"> 0.3711 </td>
        <td style="text-align:center"> 0.3426 </td>
        <td style="text-align:center"> 0.3710 </td>
        <td style="text-align:center"> 0.1914 </td>
        <td style="text-align:center"> 0.2879 </td>
        <td style="text-align:center"> <a href="https://arxiv.org/abs/2205.13504">Link</a> </td>
        <td style="text-align:center"> 2023 </td>    
    </tr>
    <tr>
        <td style="text-align:center"> NLinear </td>
        <td style="text-align:center"> 0.3924 </td>
        <td style="text-align:center"> <b>0.4001</b> </td>
        <td style="text-align:center"> <b>0.2888</b> </td>
        <td style="text-align:center"> <b>0.3375</b> </td>
        <td style="text-align:center"> 0.3486 </td>
        <td style="text-align:center"> <b>0.3695</b> </td>
        <td style="text-align:center"> <b>0.1814</b> </td>
        <td style="text-align:center"> <b>0.2630</b> </td>
        <td style="text-align:center"> <a href="https://arxiv.org/abs/2205.13504">Link</a> </td>
	    <td style="text-align:center"> 2023 </td> 
    </tr>
    <tr>
        <td style="text-align:center"> TimesNet </td>
        <td style="text-align:center"> 0.4077 </td>
        <td style="text-align:center"> 0.4218 </td>
        <td style="text-align:center"> 0.3263 </td>
        <td style="text-align:center"> 0.3688 </td>
        <td style="text-align:center"> 0.3359 </td>
        <td style="text-align:center"> 0.3761 </td>
        <td style="text-align:center"> 0.1844 </td>
        <td style="text-align:center"> 0.2643 </td>
        <td style="text-align:center"> <a href="https://arxiv.org/abs/2210.02186">Link</a> </td>
        <td style="text-align:center"> 2023 </td> 	
    </tr>
    <tr>
        <td style="text-align:center"> Crossformer </td>
        <td style="text-align:center"> 0.4479 </td>
        <td style="text-align:center"> 0.4602 </td>
        <td style="text-align:center"> 1.1278 </td>
        <td style="text-align:center"> 0.7961 </td>
        <td style="text-align:center"> 0.3772 </td>
        <td style="text-align:center"> 0.4116 </td>
        <td style="text-align:center"> 0.2976 </td>
        <td style="text-align:center"> 0.3794 </td>
        <td style="text-align:center"> <a href="https://openreview.net/forum?id=vSVLM2j9eie">Link</a> </td>
        <td style="text-align:center"> 2023 </td> 	
    </tr>
    <tr>
        <td style="text-align:center"> Crossformer<sup>[1]</sup> </td>
        <td style="text-align:center"> 0.4004 </td>
        <td style="text-align:center"> 0.4376 </td>
        <td style="text-align:center"> 1.0520 </td>
        <td style="text-align:center"> 0.7184 </td>
        <td style="text-align:center"> <b>0.3218</b> </td>
        <td style="text-align:center"> 0.3727 </td>
        <td style="text-align:center"> 0.7657 </td>
        <td style="text-align:center"> 0.6018 </td>
        <td style="text-align:center"> <a href="https://openreview.net/forum?id=vSVLM2j9eie">Link</a> </td>
        <td style="text-align:center"> 2023 </td> 	
    </tr>
</table>

[1] When we set seq_len to 672 and seg_len to 12, the performance of Crossformer is similar to the original paper. Note that the input and output of this Crossformer are already different from other algorithms, so the worth of the test results is yet to be verified.

## Develop your own model

- Add your model files to the folder `./layers/yourmodel/` and`./models/yourmodel.py`.
- Include the newly added model in the `Exp_Main.model_dict` of `./exp/exp_main.py`.
- Create the corresponding scripts under the folder `./scripts/yourmodel.sh`.

## Contact

If you have any questions or suggestions, feel free to contact:

- Chengsen Wang ([cswang@bupt.edu.cn](mailto:cswang@bupt.edu.cn))
- Jinming Wu ([kimor.wu@outlook.com](mailto:kimor.wu@outlook.com))

or describe it in Issues.

## Acknowledgement

This library is constructed based on the following repos:

- Reformer: https://github.com/lucidrains/reformer-pytorch
- Informer: https://github.com/zhouhaoyi/Informer2020
- Autoformer: https://github.com/thuml/Autoformer
- FEDformer: https://github.com/MAZiqing/FEDformer
- DLinear: https://github.com/cure-lab/LTSF-Linear
- NLinear: https://github.com/cure-lab/LTSF-Linear
- TimesNet: https://github.com/thuml/TimesNet
- Crossformer: https://github.com/Thinklab-SJTU/Crossformer
